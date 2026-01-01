# Simple Image Creator (CrewAI + Google Gemini)

Sistema **multiagentes** que gera banners corporativos com um fluxo **sequencial** em **CrewAI**:
1. **Prompt Engineer** (texto) refina o brief em um *prompt visual curto*.
2. **Image Generator** (multimodal) usa o prompt para gerar a **imagem PNG** via **Google Gemini**.

> **Resumo:** O modelo **Gemini** faz o raciocínio e retorna conteúdo (texto/imagem); quem **salva o PNG** é o **agente Image Generator** no código da crew.

---

## 📁 Estrutura do projeto

```text
crews/simple_image_creator/
├─ crew_create_image_final_fixed.py     # script principal com logs detalhados
├─ saidas/                              # arquivos de saída (.png, .log)
├─ doc/                                 # imagens de documentação/diagrama
│  ├─ crew_PE_IG.png
│  ├─ banner_20251231_152313_991837.png
│  └─ (outros visuais de apoio)
└─ .env                                 # configuração local (variáveis de ambiente)
```

---

## 🧠 Arquitetura & Fluxo

### Agentes
- **Prompt Engineer (Gemini – texto):** recebe o brief e entrega um **prompt visual** curto e objetivo (estilo, composição, paleta, iluminação, textura).
- **Image Generator (Gemini – imagem):** consome o prompt visual, chama o modelo de imagem do Gemini, extrai a mídia da resposta e **salva** o PNG com o tamanho final.

### Diagrama simplificado (pictórico)

![crew_PE_IG.png](https://github.com/raspcwalter/agenticAI/blob/main/crews/simple_image_creator/doc/crew_PE_IG.png)

**Exemplo de saída (banner gerado):**

![banner_20251231_152313_991837.png](https://github.com/raspcwalter/agenticAI/blob/main/crews/simple_image_creator/doc/banner_20251231_152313_991837.png)

---

## ⚙️ Requisitos

- **Python 3.10+**
- Pacotes:
  - `google-genai` (Google Gen AI SDK)
  - `python-dotenv`
  - `crewai` e `crewai-tools`
  - `Pillow` (PIL)
- **Chave de API do Google AI Studio**: `GEMINI_API_KEY`

> **Observação:** O SDK **Google Gen AI** suporta tanto o **Gemini Developer API** quanto **Vertex AI**. Para uso direto com o Developer API, basta configurar `GEMINI_API_KEY`. Se você também definir `GOOGLE_API_KEY`, o SDK dá **prioridade** a ela.

Referências:
- [Google Gen AI SDK](https://googleapis.github.io/python-genai/)
- [Generating Content (v1beta)](https://ai.google.dev/api/generate-content)
- [Image generation com Gemini](https://ai.google.dev/gemini-api/docs/image-generation)

---

## 🔧 Configuração

1. **Instale as dependências:**
   ```bash
   pip install -U google-genai python-dotenv "crewai[tools]" crewai-tools pillow
   ```

2. **Crie `.env` (exemplo):**
   ```ini
   # Chave do AI Studio (defina apenas UMA; prefira GEMINI_API_KEY)
   GEMINI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx

   # Modelos
   GEMINI_TEXT_MODEL=gemini-2.5-flash
   GEMINI_IMAGE_MODEL=gemini-2.5-flash-image

   # Saída
   OUTPUT_DIR=saidas
   OUTPUT_FILENAME=banner.png
   WIDTH=1024
   HEIGHT=1024

   # Logs
   LOG_LEVEL=INFO
   LOG_FILE=saidas/crew_image.log

   # Controle
   MIN_SECONDS_BETWEEN_IMAGE_CALLS=10
   ALLOW_LOCAL_RENDER_FALLBACK=true
   ```

> **Dica:** Ajuste `WIDTH` e `HEIGHT` para **1200×628** (formato de card LinkedIn) quando necessário.

---

## ▶️ Execução

```bash
python crew_create_image_final_fixed.py
```

**O que acontece:**
- A **Crew** inicia o processo sequencial (Prompt Engineer → Image Generator).
- O **Prompt Engineer** usa `models.generate_content` (texto) para produzir o **prompt visual**.
- O **Image Generator** chama `models.generate_content` com o **modelo de imagem** (ex.: `gemini-2.5-flash-image`) e extrai a imagem da resposta (via `parts` → `inline_data`/`as_image`), aplica `enforce_size` e **salva** o arquivo como **PNG** em `saidas/`.

**Logs:** verifique `saidas/crew_image.log` para tempos, tentativas, backoff (429/503), modelo usado e caminho final.

---

## 🧪 Testes rápidos

**Teste de texto (sanidade da chave):**
```bash
python - << 'PY'
from google import genai
client = genai.Client()
resp = client.models.generate_content(model="gemini-2.5-flash", contents="Ping?")
print("Texto OK:", bool(resp.text))
PY
```

**Teste de imagem (modelo de imagem Gemini):**
```bash
python - << 'PY'
from google import genai
client = genai.Client()
resp = client.models.generate_content(model="gemini-2.5-flash-image", contents=["Create a simple blue icon"])
parts = getattr(resp, "parts", None) or resp.candidates[0].content.parts
print("Imagem OK:", any(getattr(p,"inline_data",None) or hasattr(p,"as_image") for p in parts))
PY
```

---

## 📌 Observações importantes

- **Quem salva o PNG?** O **agente Image Generator** (código) — **não** a LLM. A LLM retorna os dados; o agente extrai e escreve o arquivo.
- **Disponibilidade/região:** alguns modelos/recursos podem ter **restrições regionais** ou ficar **temporariamente sobrecarregados** (503). O script já faz **retry com backoff** e tenta **preview** quando aplicável.

Referências:
- [Image generation guide](https://ai.google.dev/gemini-api/docs/image-generation)
- [Generating Content v1beta](https://ai.google.dev/api/generate-content)

---

## 📣 Uso em comunicação (LinkedIn)

- Poste o diagrama **pictórico** e, no texto, explique:
  - **Prompt Engineer (Gemini – texto)** → cria o prompt visual.
  - **Image Generator (Gemini – imagem)** → gera e salva o PNG.
- Inclua **métricas** (ex.: *PE ~10,5 s | IG ~1,6 s | 1 tentativa | PNG 1200×628 | logs em `saidas/crew_image.log`*).

---

## 🛡️ Conformidade e Segurança

- Respeite políticas de conteúdo e direitos autorais ao gerar imagens.
- Saídas de imagem dos modelos Gemini incluem **SynthID watermark** para transparência/identificação.

Referências:
- [Gemini Image generation](https://ai.google.dev/gemini-api/docs/image-generation)

---

## 🤝 Licença & Créditos

- Código sob a licença do repositório principal.
- Framework: **CrewAI** (agentes, tarefas, processos).

Referências:
- [Documentação CrewAI](https://docs.crewai.com/)

---

## 🧰 FAQ

**1) Posso usar Vertex AI em vez do Developer API?**  
Sim. O SDK `google-genai` também suporta **Vertex AI**; nesse caso, configure `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT` e `GOOGLE_CLOUD_LOCATION`.

**2) Por que a imagem às vezes não vem?**  
Pode ser **sobrecarga** do modelo (503), **restrição regional** do endpoint de imagem ou falta de cota. Use **backoff**, troque o modelo de imagem (preview) e tente novamente.

**3) Como garantir o tamanho exato?**  
O código aplica `enforce_size` e letterbox/crop quando necessário, salvando sempre o **PNG** final nas dimensões definidas.

---

> Dúvidas ou melhorias? Abra uma issue ou peça uma **variação** de diagrama (quadrado, 1200×628, 1080×1920) e paleta alinhada à sua identidade visual.
