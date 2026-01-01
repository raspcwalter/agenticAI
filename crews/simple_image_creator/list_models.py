#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lista modelos disponíveis via Google Gen AI SDK (Gemini).

Recursos:
  - Modo Developer API (default) com versão de API configurável (v1/v1beta).
  - Modo Vertex AI (--vertex) com GOOGLE_CLOUD_PROJECT/LOCATION.
  - Filtros por ação (--action generateContent) e apenas imagem (--image-only, heurístico).
  - Saída em texto legível ou JSON (--json).
  - Limite de itens (--limit).

Pré-requisitos:
  pip install -U google-genai python-dotenv
  Defina GEMINI_API_KEY (Developer API) OU GOOGLE_CLOUD_PROJECT/GOOGLE_CLOUD_LOCATION (Vertex).

Referências:
  - SDK (google.genai): Client, models.list/get
    https://googleapis.github.io/python-genai/
  - API referencia (models.list / models.get)
    https://ai.google.dev/api/models
  - Overview SDK + Vertex AI
    https://docs.cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview
  - Geração de imagem (retorno via response.parts)
    https://ai.google.dev/gemini-api/docs/image-generation
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any

from dotenv import load_dotenv

from google import genai
from google.genai import types


def build_client(args: argparse.Namespace) -> genai.Client:
    """
    Constrói o Client do Google Gen AI SDK conforme flags/ambiente.
    """
    if args.vertex:
        project = args.project or os.getenv("GOOGLE_CLOUD_PROJECT")
        location = args.location or os.getenv("GOOGLE_CLOUD_LOCATION")
        if not project or not location:
            print("[ERRO] Para --vertex, informe --project e --location ou defina GOOGLE_CLOUD_PROJECT/GOOGLE_CLOUD_LOCATION.", file=sys.stderr)
            sys.exit(2)
        # Vertex AI
        return genai.Client(vertexai=True, project=project, location=location)
    else:
        # Developer API com versão de API
        api_version = args.api_version or "v1"
        return genai.Client(http_options=types.HttpOptions(api_version=api_version))


def is_image_model_heuristic(model_name: str) -> bool:
    """
    Heurística para marcar modelos 'image-capable' sem chamar a API de geração.
    Observações:
      - Modelos com 'image' no nome são fortes candidatos (ex.: 'gemini-2.5-flash-image').
      - Em alguns projetos, 'gemini-2.0-flash' pode ter saída de imagem experimental.

    A lista efetiva pode variar por serviço/tier/região; sempre confira via models.list().
    """
    name = model_name.lower()
    if "image" in name:
        return True
    # Caso especial conhecido para alguns projetos:
    if name.endswith("gemini-2.0-flash") or name.endswith("/gemini-2.0-flash"):
        return True
    return False


def collect_models(client: genai.Client, action_filter: str | None = None, limit: int | None = None) -> List[Dict[str, Any]]:
    """
    Percorre models.list() e retorna uma lista com metadados essenciais:
      - name (resource: models/{id})
      - display_name (quando disponível)
      - supported_actions
      - is_image_model (heurístico)
    """
    out: List[Dict[str, Any]] = []
    count = 0
    for m in client.models.list():
        # Em SDK novo, 'supported_actions' lista as ações (ex.: 'generateContent', 'embedContent')
        actions = getattr(m, "supported_actions", []) or []
        if action_filter and (action_filter not in actions):
            continue

        name = getattr(m, "name", "")
        display_name = getattr(m, "display_name", "")
        short_name = name.split("/")[-1] if isinstance(name, str) else str(name)

        item = {
            "name": name,
            "id": short_name,
            "display_name": display_name,
            "supported_actions": actions,
            "is_image_model": is_image_model_heuristic(short_name),
        }
        out.append(item)
        count += 1
        if limit and count >= limit:
            break
    return out


def print_models_human(models: List[Dict[str, Any]]):
    """
    Saída legível (texto).
    """
    if not models:
        print("Nenhum modelo encontrado para os filtros aplicados.")
        return

    print(f"Modelos encontrados: {len(models)}\n")
    for i, m in enumerate(models, start=1):
        actions = ", ".join(m.get("supported_actions") or [])
        im_flag = " (image-capable: heurístico)" if m.get("is_image_model") else ""
        disp = f" — {m['display_name']}" if m.get("display_name") else ""
        print(f"{i:02d}. {m['id']}{disp}{im_flag}")
        print(f"     resource: {m['name']}")
        print(f"     actions : {actions or '—'}")
        print()


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Lista modelos do Gemini via Google Gen AI SDK (Developer API/Vertex AI)."
    )
    parser.add_argument("--vertex", action="store_true", help="Usa Vertex AI (requer --project/--location ou variáveis de ambiente).")
    parser.add_argument("--project", type=str, help="ID do projeto GCP (apenas Vertex).")
    parser.add_argument("--location", type=str, help="Local (ex.: us-central1) (apenas Vertex).")
    parser.add_argument("--api-version", type=str, default="v1", help="Versão de API (Developer API). Ex.: v1 ou v1beta. Padrão: v1.")
    parser.add_argument("--action", type=str, default=None, help="Filtra por ação suportada (ex.: generateContent, embedContent).")
    parser.add_argument("--image-only", action="store_true", help="Mostra apenas modelos marcados como image-capable (heurístico).")
    parser.add_argument("--limit", type=int, default=None, help="Limita a quantidade listada.")
    parser.add_argument("--json", action="store_true", help="Saída em JSON ao invés de texto legível.")
    args = parser.parse_args()

    try:
        client = build_client(args)
    except Exception as e:
        print(f"[ERRO] Falha ao construir o client: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        models = collect_models(client, action_filter=args.action, limit=args.limit)
    except Exception as e:
        print(f"[ERRO] Falha ao listar modelos: {e}", file=sys.stderr)
        sys.exit(3)

    if args.image_only:
        models = [m for m in models if m.get("is_image_model")]

    if args.json:
        print(json.dumps(models, ensure_ascii=False, indent=2))
    else:
        print_models_human(models)


if __name__ == "__main__":
    main()
