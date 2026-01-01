#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Crew final (FIX): Gemini 2.5 Flash Image + logs detalhados + fallback local estável (Pillow)
# Usa client default (v1beta implícito), corrige escapes de 
# em strings literais (evita SyntaxError).

import os, io, time, base64, traceback, logging
from datetime import datetime
from typing import Type, List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from PIL import Image, ImageDraw, ImageFont

def _setup_logger() -> logging.Logger:
    load_dotenv()
    log_level = os.getenv('LOG_LEVEL','INFO').upper()
    log_file = os.getenv('LOG_FILE','saidas/crew_image.log')
    logger = logging.getLogger('crew_image')
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s','%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(); ch.setFormatter(fmt); logger.addHandler(ch)
    try: os.makedirs(os.path.dirname(log_file), exist_ok=True)
    except Exception: pass
    fh = logging.FileHandler(log_file, encoding='utf-8'); fh.setFormatter(fmt); logger.addHandler(fh)
    logger.info('Logger inicializado (level=%s, file=%s)', log_level, log_file)
    return logger

logger = _setup_logger()

MIN_SECONDS_BETWEEN_IMAGE_CALLS = int(os.getenv('MIN_SECONDS_BETWEEN_IMAGE_CALLS','10'))
_last_image_call_ts = 0.0
ALLOW_LOCAL_RENDER_FALLBACK = (os.getenv('ALLOW_LOCAL_RENDER_FALLBACK','true').lower()=='true')

load_dotenv()
client = genai.Client()
logger.info('Client inicializado (API default / v1beta).')

GEMINI_TEXT_MODEL = os.getenv('GEMINI_TEXT_MODEL','gemini-2.5-flash')
GEMINI_IMAGE_MODEL = os.getenv('GEMINI_IMAGE_MODEL','gemini-2.5-flash-image')
logger.info('Modelos: texto=%s, imagem=%s', GEMINI_TEXT_MODEL, GEMINI_IMAGE_MODEL)

DEFAULT_OUTPUT_DIR = os.getenv('OUTPUT_DIR','saidas')
DEFAULT_OUTPUT_FILENAME = os.getenv('OUTPUT_FILENAME','banner.png')
DEFAULT_WIDTH = int(os.getenv('WIDTH','1024'))
DEFAULT_HEIGHT = int(os.getenv('HEIGHT','1024'))
logger.info('Saída: dir=%s, filename=%s, size=%dx%d', DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_FILENAME, DEFAULT_WIDTH, DEFAULT_HEIGHT)

MAX_ATTEMPTS = 3
IMAGE_MODEL_CANDIDATES: List[str] = [GEMINI_IMAGE_MODEL, 'gemini-2.5-flash-image-preview']
logger.info('Candidatos de imagem: %s', IMAGE_MODEL_CANDIDATES)

from pydantic import BaseModel, Field
class PromptInput(BaseModel):
    brief: str = Field(..., description='Brief visual para refinar em 1–3 frases.')
class ImageInput(BaseModel):
    image_description: str = Field(..., description='Prompt final para gerar a imagem.')
    width: int = Field(default=DEFAULT_WIDTH, description='Largura px.')
    height: int = Field(default=DEFAULT_HEIGHT, description='Altura px.')
    enforce_size: bool = Field(default=True, description='Garante exatamente width x height via pós-processamento.')

def _error_code(e: ClientError) -> Optional[int]:
    for attr in ('code','status_code'):
        try:
            val = getattr(e, attr, None)
            if val is not None:
                return int(val)
        except Exception:
            pass
    try:
        return int(e.response_json.get('error',{}).get('code'))
    except Exception:
        return None

def _retry_delay_seconds(e: Exception, default: int = 10) -> int:
    if isinstance(e, ClientError):
        try:
            details = e.response_json.get('error',{}).get('details',[])
            for d in details:
                if d.get('@type','').endswith('RetryInfo'):
                    retry = d.get('retryDelay', f'{default}s').strip().lower()
                    if retry.endswith('s'):
                        return int(float(retry[:-1]))
        except Exception:
            pass
    return default

def _rate_gate():
    global _last_image_call_ts
    now = time.monotonic()
    wait = (_last_image_call_ts + MIN_SECONDS_BETWEEN_IMAGE_CALLS) - now
    if wait > 0:
        logger.debug('Rate gate: aguardando %.2fs', wait)
        time.sleep(wait)
    _last_image_call_ts = time.monotonic()

def _ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
        logger.debug('Dir ok: %s', path)
    except Exception as e:
        logger.warning('Criar dir %s: %s', path, e)

def _extract_image_from_part(part) -> Optional[Image.Image]:
    try:
        if hasattr(part, 'as_image'):
            maybe_img = part.as_image()
            if isinstance(maybe_img, Image.Image):
                logger.debug('as_image(): %dx%d', maybe_img.width, maybe_img.height)
                return maybe_img
    except Exception as ex_as:
        logger.debug('as_image() falhou: %s', ex_as)
    try:
        inline = getattr(part, 'inline_data', None)
        data = getattr(inline, 'data', None) if inline else None
        if not data:
            logger.debug('Sem inline_data.')
            return None
        raw = data if isinstance(data,(bytes,bytearray)) else base64.b64decode(data)
        img = Image.open(io.BytesIO(raw))
        logger.debug('inline_data OK: %dx%d', img.width, img.height)
        return img
    except Exception as ex_inline:
        logger.debug('inline_data falhou: %s', ex_inline)
        return None

def _save_png(img: Image.Image, target_path: str):
    img.save(target_path, format='PNG')
    logger.info('Salvo: %s (%dx%d)', target_path, img.width, img.height)
    return target_path

def _letterbox_to_exact(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    bg = Image.new('RGBA', (target_w, target_h), (0, 0, 0, 0))
    img_ratio = img.width / img.height
    tgt_ratio = target_w / target_h
    if img_ratio > tgt_ratio:
        new_w = target_w; new_h = int(new_w / img_ratio)
    else:
        new_h = target_h; new_w = int(new_h * img_ratio)
    logger.debug('Letterbox: %dx%d -> %dx%d', img.width, img.height, new_w, new_h)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    offset = ((target_w - new_w)//2, (target_h - new_h)//2)
    bg.paste(resized, offset)
    return bg

def _enforce_exact_size(img: Image.Image, w: int, h: int) -> Image.Image:
    if img.width == w and img.height == h:
        logger.debug('Tamanho exato já %dx%d', w, h)
        return img
    logger.debug('Aplicando letterbox para %dx%d', w, h)
    return _letterbox_to_exact(img, w, h)

def _unique_filename(base_name: str) -> str:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root, ext = os.path.splitext(base_name)
    return f"{root}_{ts}{ext or '.png'}"

def _local_render_banner(width: int, height: int, title_text: str = 'Relatório 2026') -> Image.Image:
    try:
        logger.warning('Fallback local (Pillow) acionado.')
        bg = (26, 47, 75, 255); white = (255, 255, 255, 255)
        img = Image.new('RGBA', (width, height), bg)
        draw = ImageDraw.Draw(img); font_title = ImageFont.load_default()
        bbox = draw.textbbox((0,0), title_text, font=font_title)
        title_w = bbox[2]-bbox[0]; title_h = bbox[3]-bbox[1]
        title_x = (width - title_w)//2; title_y = height//3 - title_h//2
        draw.text((title_x, title_y), title_text, font=font_title, fill=white)
        bar_w = max(6, width//80); spacing = bar_w
        base_y = title_y + title_h + (height//20); base_x = (width//2) - (2*bar_w + spacing)
        heights = [height//14, height//10, height//8]
        for i, h in enumerate(heights):
            x0 = base_x + i*(bar_w + spacing); y0 = base_y - h
            x1 = x0 + bar_w; y1 = base_y
            draw.rectangle([(x0,y0),(x1,y1)], fill=white)
        shield_w = bar_w*5; shield_h = height//12
        sx = base_x + 3*(bar_w + spacing) + spacing*2; sy = base_y - shield_h
        shield_points = [(sx,sy),(sx+shield_w,sy),(sx+shield_w-bar_w,sy+shield_h//2),(sx+shield_w//2,sy+shield_h),(sx+bar_w,sy+shield_h//2)]
        draw.polygon(shield_points, fill=white)
        logger.info('Fallback local renderizado (%dx%d).', width, height)
        return img
    except Exception as e:
        logger.error('Falha no fallback local, quadrado azul: %s', e)
        return Image.new('RGBA', (width, height), (26,47,75,255))

class GeminiPromptTool(BaseTool):
    name: str = 'Gemini Prompt Engineer'
    description: str = ('Refina o brief visual em 1–3 frases objetivas; inclui estilo, composição, paleta, iluminação e textura.')
    args_schema: Type[BaseModel] = PromptInput
    model: str = GEMINI_TEXT_MODEL
    def _run(self, brief: str) -> str:
        logger.info('PromptTool: refinando brief.')
        t0 = time.time()
        try:
            resp = client.models.generate_content(model=self.model, contents=brief)
            refined = (getattr(resp,'text','') or '').strip()
            logger.info('PromptTool: OK em %.1fms.', (time.time()-t0)*1000)
            return refined or brief.strip()
        except Exception as e:
            logger.error('PromptTool: ERRO em %.1fms: %s', (time.time()-t0)*1000, e)
            return f"[ERRO Gemini Prompt] {e}"

class GeminiImageTool(BaseTool):
    name: str = 'Gemini Image Generator'
    description: str = 'Gera a imagem com Gemini 2.5 (ou local) e salva em PNG.'
    args_schema: Type[BaseModel] = ImageInput
    output_dir: str = DEFAULT_OUTPUT_DIR
    output_filename: str = DEFAULT_OUTPUT_FILENAME
    def _try_gemini_image(self, image_description: str, width: int, height: int) -> Optional[Image.Image]:
        _rate_gate()
        prompt = f"{image_description}\nDesired output: {width}x{height} px, PNG."
        logger.info('ImageTool: tentando Gemini (2.5 image/preview).')
        for model_id in IMAGE_MODEL_CANDIDATES:
            logger.info('ImageTool: model_id=%s', model_id)
            attempts = 0
            while attempts < MAX_ATTEMPTS:
                attempts += 1
                t0 = time.time()
                try:
                    resp = client.models.generate_content(model=model_id, contents=[prompt])
                    duration_ms = (time.time()-t0)*1000
                    sdk_resp = getattr(resp,'sdk_http_response',None)
                    status = getattr(sdk_resp,'status_code','N/A')
                    logger.info('ImageTool: resposta status=%s, tempo=%.1fms (attempt=%d/%d)', status, duration_ms, attempts, MAX_ATTEMPTS)
                    parts = getattr(resp,'parts',None)
                    if not parts:
                        try:
                            parts = resp.candidates[0].content.parts
                        except Exception:
                            parts = None
                    logger.debug('ImageTool: parts_found=%s', bool(parts))
                    if not parts:
                        logger.warning('ImageTool: sem parts; próximo modelo.')
                        break
                    for p in parts:
                        has_inline = getattr(p,'inline_data',None) is not None
                        has_as_image = hasattr(p,'as_image')
                        logger.debug('Part: inline=%s, as_image=%s', has_inline, has_as_image)
                        img = _extract_image_from_part(p)
                        if isinstance(img, Image.Image):
                            logger.info('ImageTool: imagem extraída (%dx%d).', img.width, img.height)
                            return img
                    logger.warning('ImageTool: parts sem imagem; próximo modelo.')
                    break
                except ClientError as e:
                    code = _error_code(e)
                    msg = e.response_json.get('error',{}).get('message') if hasattr(e,'response_json') else str(e)
                    logger.error('ImageTool: ClientError code=%s, msg=%s', code, msg)
                    if code in (429,503):
                        delay = _retry_delay_seconds(e, default=10)
                        logger.warning('ImageTool: backoff=%ds (attempt=%d/%d)', delay, attempts, MAX_ATTEMPTS)
                        time.sleep(delay)
                        continue
                    break
                except Exception as ex:
                    logger.error('ImageTool: exceção: %s', ex)
                    delay = _retry_delay_seconds(ex, default=6)
                    logger.warning('ImageTool: pequeno backoff=%ds e nova tentativa.', delay)
                    time.sleep(delay)
                    continue
        return None

    def _run(self, image_description: str, width: int, height: int, enforce_size: bool = True) -> str:
        logger.info("ImageTool: start desc='%s' size=%dx%d", image_description, width, height)
        t0 = time.time()
        try:
            _ensure_dir(self.output_dir)
            unique_name = _unique_filename(self.output_filename)
            target = os.path.join(self.output_dir, os.path.basename(unique_name))
            logger.debug('Arquivo destino: %s', target)
            img = self._try_gemini_image(image_description, width, height)
            if img is None and ALLOW_LOCAL_RENDER_FALLBACK:
                img = _local_render_banner(width, height, title_text='Relatório 2026')
            if img is None:
                logger.error('ImageTool: falha — nem Gemini nem fallback local renderizaram.')
                return '[ERRO] Nenhuma imagem foi gerada (Gemini) e fallback local desativado.'
            if enforce_size:
                before = (img.width, img.height)
                img = _enforce_exact_size(img, width, height)
                after = (img.width, img.height)
                logger.info('ImageTool: enforce_size %s -> %s', before, after)
            _save_png(img, target)
            logger.info('ImageTool: concluído em %.1fms; path=%s', (time.time()-t0)*1000, target)
            return target
        except ClientError as e:
            code = _error_code(e)
            dj = {}
            try:
                dj = e.response_json
            except Exception:
                dj = {'error': {'message': str(e)}}
            logger.error('ImageTool: ClientError code=%s json=%s', code, dj)
            return f"[ERRO Gemini] {code or 'N/A'} {dj}"
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            logger.error('ImageTool: exceção geral: %s\n%s', e, tb)
            try:
                _ensure_dir(self.output_dir)
                target = os.path.join(self.output_dir, os.path.basename(_unique_filename(DEFAULT_OUTPUT_FILENAME)))
                img = Image.new('RGBA', (width, height), (26,47,75,255))
                _save_png(img, target)
                logger.warning('ImageTool: fallback mínimo aplicado; path=%s', target)
                return target
            except Exception as e2:
                return f"[ERRO] {e}\n{tb}\n[FALHA Fallback mínimo: {e2}]"

def build_crew() -> Crew:
    logger.info('Build crew: iniciando.')
    prompt_tool = GeminiPromptTool()
    image_tool = GeminiImageTool()
    llm_text = LLM(model=f"gemini/{GEMINI_TEXT_MODEL}", api_key=os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'), temperature=0.7)
    llm_image_agent = LLM(model=f"gemini/{GEMINI_TEXT_MODEL}", api_key=os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'), temperature=0.2)
    prompt_agent = Agent(role='Prompt Engineer', goal='Transformar requisitos em prompt visual otimizado.', backstory='Profissional de comunicação visual focado em clareza e estética.', tools=[prompt_tool], llm=llm_text, verbose=True, allow_delegation=False, max_iterations=1)
    image_agent = Agent(role='Image Generator', goal='Gerar a imagem final com qualidade e dimensões especificadas.', backstory='Artista digital com APIs multimodais e sensibilidade de design.', tools=[image_tool], llm=llm_image_agent, verbose=True, allow_delegation=False, max_iterations=1)
    task_prompt = Task(description=('Refine o brief do usuário em 1–3 frases: {brief}. Inclua estilo, composição, paleta de cores, iluminação e textura. Evite marcas.'), expected_output='Um prompt visual curto e objetivo.', agent=prompt_agent)
    task_render = Task(description=('Usar o prompt gerado para criar a imagem {width}×{height} em PNG e salvar no disco.'), expected_output='Caminho do arquivo PNG gerado.', agent=image_agent)
    crew = Crew(agents=[prompt_agent, image_agent], tasks=[task_prompt, task_render], verbose=True)
    logger.info('Build crew: pronto.')
    return crew

def run():
    load_dotenv()
    logger.info('Execução: run()')
    inputs = {
        'brief': ('Banner 1200x628 com fundo azul escuro, título \'Relatório 2026\', ícones minimalistas de gráfico e segurança; estilo clean corporativo.'),
        'width': DEFAULT_WIDTH,
        'height': DEFAULT_HEIGHT,
    }
    crew = build_crew()
    result = crew.kickoff(inputs=inputs)
    logger.info('Resultado final: %s', result)
    print('\n=== Resultado ===')
    print(result)

if __name__ == '__main__':
    try: os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    except Exception: pass
    run()
