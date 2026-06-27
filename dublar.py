#!/usr/bin/env python3
"""
Dublar 2.0 — Pipeline premium de dublagem de vídeos EN → PT-BR

Usa Whisper (transcrição) + GPT-4o-mini (tradução contextual) + OpenAI TTS (voz natural)
para dublar vídeos em inglês para português brasileiro com alta qualidade.

Uso:
    python3 dublar.py video.mov
    python3 dublar.py video.mp4 --modelo small --agrupar 30
    python3 dublar.py video.mp4 --apenas-legendas
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import whisper
except ImportError:
    print("❌ Whisper não instalado. Execute: pip install openai-whisper")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("❌ OpenAI não instalado. Execute: pip install openai")
    sys.exit(1)

try:
    import pysrt
except ImportError:
    print("❌ pysrt não instalado. Execute: pip install pysrt")
    sys.exit(1)


# ============================================================
# Configuração
# ============================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Voz OpenAI TTS — opções: alloy, echo, fable, onyx, nova, shimmer
OPENAI_TTS_VOICE = "onyx"
OPENAI_TTS_MODEL = "tts-1"  # tts-1 (rápido) ou tts-1-hd (qualidade máxima)

# Modelo de tradução
TRANSLATION_MODEL = "gpt-4o-mini"

# System prompt para tradução contextual de cursos técnicos
TRANSLATION_SYSTEM_PROMPT = """Você é um tradutor especializado em cursos de programação e tecnologia.

Regras:
1. Traduza de inglês para português do Brasil de forma natural e fluida.
2. Mantenha termos técnicos em inglês quando são usados assim no Brasil:
   - Nomes de linguagens/frameworks: React, JavaScript, TypeScript, Python, Vue, Angular, Node.js
   - Conceitos de programação: array, string, boolean, callback, promise, async, await, middleware, deploy, commit, push, pull, merge, branch, hook, state, props, component, render
   - Ferramentas: npm, yarn, webpack, Docker, Git, GitHub, VS Code, terminal
   - Siglas: API, REST, GraphQL, HTTP, JSON, CSS, HTML, DOM, SQL, CLI, SDK, CDN
3. Traduza como se fosse um professor brasileiro explicando o conceito — use linguagem acessível.
4. Não adicione explicações extras. Traduza fielmente o conteúdo original.
5. Mantenha a mesma estrutura e ordem das frases.
6. Nunca traduza nomes de funções, variáveis, ou código-fonte mencionados na fala.
7. Use "a gente" em vez de "nós" quando soar mais natural.
8. Para termos como "we're going to", traduza como "vamos" (não "iremos").
"""


# ============================================================
# Etapa 1: Extrair áudio do vídeo
# ============================================================

def extrair_audio(video_path: Path, output_path: Path) -> Path:
    """Extrai o áudio do vídeo como WAV mono 16kHz."""
    print(f"\n🎵 Extraindo áudio de: {video_path.name}")

    audio_path = output_path / "audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        str(audio_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Erro ao extrair áudio: {result.stderr}")
        sys.exit(1)

    print(f"   ✅ Áudio extraído")
    return audio_path


# ============================================================
# Etapa 2: Transcrever com Whisper
# ============================================================

def transcrever(audio_path: Path, modelo: str) -> list[dict]:
    """Transcreve o áudio usando Whisper."""
    print(f"\n📝 Transcrevendo com Whisper (modelo: {modelo})...")
    print("   Isso pode levar alguns minutos...")

    model = whisper.load_model(modelo)
    result = model.transcribe(
        str(audio_path),
        language="en",
        verbose=False,
        word_timestamps=False,
    )

    segmentos = []
    for seg in result["segments"]:
        segmentos.append({
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        })

    total_segs = len(segmentos)
    duracao = segmentos[-1]["end"] if segmentos else 0
    print(f"   ✅ Transcrição: {total_segs} segmentos, {duracao:.1f}s")
    return segmentos


# ============================================================
# Etapa 2.5: Agrupar segmentos
# ============================================================

def agrupar_segmentos(segmentos: list[dict], duracao_max: float) -> list[dict]:
    """Agrupa segmentos consecutivos em blocos maiores."""
    print(f"\n🔗 Agrupando segmentos em blocos de até {duracao_max}s...")

    grupos = []
    grupo_atual = None

    for seg in segmentos:
        if grupo_atual is None:
            grupo_atual = {
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
        else:
            duracao_grupo = seg["end"] - grupo_atual["start"]
            if duracao_grupo <= duracao_max:
                grupo_atual["end"] = seg["end"]
                grupo_atual["text"] += " " + seg["text"]
            else:
                grupos.append(grupo_atual)
                grupo_atual = {
                    "id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }

    if grupo_atual is not None:
        grupos.append(grupo_atual)

    for i, grupo in enumerate(grupos):
        grupo["id"] = i

    print(f"   ✅ {len(segmentos)} segmentos → {len(grupos)} blocos")
    return grupos


# ============================================================
# Etapa 3: Traduzir com GPT-4o-mini
# ============================================================

def traduzir(segmentos: list[dict], client: OpenAI) -> list[dict]:
    """Traduz segmentos usando GPT-4o-mini com contexto técnico."""
    if not OPENAI_API_KEY:
        print("\n❌ OPENAI_API_KEY não configurada!")
        print("   1. Crie conta em: https://platform.openai.com/api-keys")
        print("   2. Export: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    print(f"\n🌐 Traduzindo {len(segmentos)} segmentos com {TRANSLATION_MODEL}...")

    segmentos_traduzidos = []

    # Traduzir em lotes de 5 segmentos por chamada (melhor contexto)
    BATCH_SIZE = 5

    for i in range(0, len(segmentos), BATCH_SIZE):
        batch = segmentos[i:i + BATCH_SIZE]

        # Montar o texto pra tradução com marcadores de segmento
        textos_numerados = []
        for j, seg in enumerate(batch):
            textos_numerados.append(f"[{j+1}] {seg['text']}")

        texto_para_traduzir = "\n".join(textos_numerados)

        user_prompt = f"""Traduza os seguintes segmentos de um curso de programação para português do Brasil.
Mantenha a numeração [1], [2], etc. Cada segmento deve ser traduzido separadamente.

{texto_para_traduzir}"""

        try:
            response = client.chat.completions.create(
                model=TRANSLATION_MODEL,
                messages=[
                    {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            traducao_raw = response.choices[0].message.content
        except Exception as e:
            print(f"   ❌ Erro na tradução: {e}")
            sys.exit(1)

        # Parsear a resposta — extrair cada segmento traduzido
        traducoes = parsear_traducao(traducao_raw, len(batch))

        for j, seg in enumerate(batch):
            texto_pt = traducoes[j] if j < len(traducoes) else seg["text"]
            segmentos_traduzidos.append({
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "text_en": seg["text"],
                "text_pt": texto_pt,
            })

        progresso = min(i + BATCH_SIZE, len(segmentos))
        print(f"   Traduzido: {progresso}/{len(segmentos)} segmentos")

    print(f"   ✅ Tradução concluída")
    return segmentos_traduzidos


def parsear_traducao(texto_raw: str, expected_count: int) -> list[str]:
    """Extrai segmentos traduzidos da resposta do GPT."""
    import re

    linhas = texto_raw.strip().split("\n")
    traducoes = []
    current_text = ""
    current_idx = -1

    for linha in linhas:
        # Tentar match com [N] no início da linha
        match = re.match(r'^\[(\d+)\]\s*(.*)', linha)
        if match:
            # Salvar o anterior
            if current_idx >= 0 and current_text:
                traducoes.append(current_text.strip())
            current_idx = int(match.group(1))
            current_text = match.group(2)
        else:
            # Continuação do segmento atual
            if current_idx >= 0:
                current_text += " " + linha.strip()

    # Salvar o último
    if current_idx >= 0 and current_text:
        traducoes.append(current_text.strip())

    # Se não conseguiu parsear, retorna o texto inteiro como um segmento
    if not traducoes:
        traducoes = [texto_raw.strip()]

    # Preencher se faltou algum
    while len(traducoes) < expected_count:
        traducoes.append("")

    return traducoes


# ============================================================
# Etapa 4: Gerar áudio com OpenAI TTS
# ============================================================

def gerar_audio_openai(texto: str, output_file: Path, client: OpenAI, voice: str, model: str):
    """Gera áudio usando OpenAI TTS API."""
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=texto,
        response_format="mp3",
    )
    response.stream_to_file(str(output_file))


def obter_duracao_audio(audio_file: str) -> float:
    """Retorna a duração de um arquivo de áudio em segundos."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json", str(audio_file)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    data = json.loads(result.stdout)
    return float(data.get("format", {}).get("duration", 0))


def ajustar_velocidade_audio(audio_file: str, duracao_alvo: float,
                              output_path: Path, seg_id: int) -> str:
    """
    Ajusta velocidade do áudio para caber na duração alvo.
    Acelera se muito longo, desacelera se muito curto.
    Limites: 0.75x a 1.5x.
    """
    duracao_atual = obter_duracao_audio(audio_file)

    if duracao_atual <= 0 or duracao_alvo <= 0:
        return audio_file

    fator = duracao_atual / duracao_alvo

    # Diferença insignificante — não ajustar
    if 0.95 <= fator <= 1.05:
        return audio_file

    # Limites pra manter qualidade da voz OpenAI
    MIN_SPEED = 0.75
    MAX_SPEED = 1.5
    fator = max(MIN_SPEED, min(MAX_SPEED, fator))

    audio_ajustado = str(output_path / f"seg_speed_{seg_id}.mp3")
    cmd = [
        "ffmpeg", "-y", "-i", audio_file,
        "-filter:a", f"atempo={fator}",
        "-vn", audio_ajustado
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return audio_file

    return audio_ajustado


def gerar_audio_tts(segmentos: list[dict], output_path: Path,
                    client: OpenAI, voice: str, model: str) -> list[dict]:
    """Gera áudio TTS para todos os segmentos, ajustando velocidade."""
    print(f"\n🔊 Gerando áudio com OpenAI TTS (voz: {voice}, modelo: {model})...")
    print(f"   Sincronizando velocidade com timestamps originais...")

    tts_dir = output_path / "tts_segments"
    tts_dir.mkdir(exist_ok=True)

    segmentos_com_audio = []
    ajustados = 0

    for i, seg in enumerate(segmentos):
        audio_file = tts_dir / f"seg_{i:04d}.mp3"

        try:
            gerar_audio_openai(seg["text_pt"], audio_file, client, voice, model)
        except Exception as e:
            print(f"   ⚠️  Erro no segmento {i}: {e}")
            continue

        # Ajustar velocidade para caber no espaço do segmento original
        duracao_disponivel = seg["end"] - seg["start"]
        audio_final = ajustar_velocidade_audio(
            str(audio_file), duracao_disponivel, tts_dir, i
        )

        if audio_final != str(audio_file):
            ajustados += 1

        segmentos_com_audio.append({
            **seg,
            "audio_file": audio_final,
        })

        if (i + 1) % 10 == 0:
            print(f"   Gerado: {i + 1}/{len(segmentos)} segmentos")

    print(f"   ✅ Áudio gerado: {len(segmentos_com_audio)} segmentos"
          f" ({ajustados} ajustados em velocidade)")
    return segmentos_com_audio


# ============================================================
# Etapa 5: Montar vídeo final
# ============================================================

def montar_audio_final(segmentos: list[dict], output_path: Path,
                       duracao_video: float) -> Path:
    """Monta o áudio final posicionando cada segmento no timestamp original."""
    print(f"\n🎬 Montando áudio final sincronizado...")

    audio_final_path = output_path / "audio_final.wav"
    concat_list = output_path / "concat.txt"

    if not segmentos:
        print("   ⚠️  Nenhum segmento de áudio para montar")
        silencio_path = output_path / "silencio.wav"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-t", str(duracao_video),
            "-acodec", "pcm_s16le", str(silencio_path)
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        return silencio_path

    with open(concat_list, "w") as f:
        last_end = 0.0

        for seg in segmentos:
            gap = seg["start"] - last_end
            if gap > 0.01:
                silence_file = output_path / f"silence_{seg['id']}.wav"
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                    "-t", str(gap),
                    "-acodec", "pcm_s16le", str(silence_file)
                ]
                subprocess.run(cmd, capture_output=True, text=True)
                f.write(f"file '{silence_file}'\n")

            seg_wav = output_path / f"seg_wav_{seg['id']}.wav"
            cmd = [
                "ffmpeg", "-y", "-i", seg["audio_file"],
                "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
                str(seg_wav)
            ]
            subprocess.run(cmd, capture_output=True, text=True)
            f.write(f"file '{seg_wav}'\n")

            last_end = seg["end"]

    # Concatenar
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-acodec", "pcm_s16le", "-ar", "44100",
        str(audio_final_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"   ❌ Erro ao concatenar: {result.stderr[:300]}")
        sys.exit(1)

    # Pad com silêncio se necessário
    duracao_audio = obter_duracao_audio(str(audio_final_path))
    if duracao_audio < duracao_video - 0.5:
        audio_padded = output_path / "audio_final_padded.wav"
        pad = duracao_video - duracao_audio
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_final_path),
            "-af", f"apad=pad_dur={pad}",
            "-acodec", "pcm_s16le", "-ar", "44100",
            str(audio_padded)
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        audio_final_path = audio_padded

    print(f"   ✅ Áudio final montado ({duracao_audio:.1f}s)")
    return audio_final_path


def montar_video_final(video_path: Path, audio_path: Path, output_video: Path):
    """Combina o vídeo original com o novo áudio dublado."""
    print(f"\n📹 Montando vídeo final...")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_video)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"   ❌ Erro ao montar vídeo: {result.stderr[:200]}")
        sys.exit(1)

    print(f"   ✅ Vídeo dublado salvo: {output_video}")


# ============================================================
# Etapa extra: Gerar legendas .srt
# ============================================================

def gerar_srt(segmentos: list[dict], output_srt: Path):
    """Gera arquivo de legendas .srt."""
    print(f"\n📄 Gerando legendas: {output_srt.name}")

    subs = pysrt.SubRipFile()

    for seg in segmentos:
        start_time = pysrt.SubRipTime(seconds=seg["start"])
        end_time = pysrt.SubRipTime(seconds=seg["end"])
        item = pysrt.SubRipItem(
            index=seg["id"] + 1,
            start=start_time,
            end=end_time,
            text=seg["text_pt"],
        )
        subs.append(item)

    subs.save(str(output_srt), encoding="utf-8")
    print(f"   ✅ Legendas salvas: {output_srt}")


# ============================================================
# Utilitários
# ============================================================

def obter_duracao_video(video_path: Path) -> float:
    """Retorna a duração do vídeo em segundos."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json", str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def verificar_dependencias():
    """Verifica se FFmpeg está instalado."""
    if not shutil.which("ffmpeg"):
        print("❌ FFmpeg não encontrado! Instale com: brew install ffmpeg")
        sys.exit(1)
    if not shutil.which("ffprobe"):
        print("❌ ffprobe não encontrado! Instale com: brew install ffmpeg")
        sys.exit(1)


def estimar_custo(segmentos: list[dict]) -> dict:
    """Estima o custo da tradução + TTS baseado no texto."""
    total_chars = sum(len(seg["text"]) for seg in segmentos)
    total_chars_pt_est = int(total_chars * 1.2)  # português ~20% mais longo

    # GPT-4o-mini: ~$0.15/1M input tokens, ~$0.60/1M output tokens
    # Estimativa: 1 token ≈ 4 chars
    tokens_in = total_chars / 4
    tokens_out = total_chars_pt_est / 4
    custo_traducao = (tokens_in * 0.15 / 1_000_000) + (tokens_out * 0.60 / 1_000_000)

    # OpenAI TTS: $15/1M chars (tts-1) ou $30/1M chars (tts-1-hd)
    custo_tts = total_chars_pt_est * 15 / 1_000_000

    return {
        "chars_en": total_chars,
        "chars_pt_est": total_chars_pt_est,
        "custo_traducao": custo_traducao,
        "custo_tts": custo_tts,
        "custo_total": custo_traducao + custo_tts,
    }


# ============================================================
# Pipeline principal
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dublar 2.0 — Dublagem premium de vídeos EN → PT-BR"
    )
    parser.add_argument("video", help="Caminho do vídeo a ser dublado")
    parser.add_argument("--idioma", default="PT-BR",
                        help="Idioma alvo (padrão: PT-BR)")
    parser.add_argument("--modelo", default="base",
                        help="Modelo Whisper: tiny, base, small, medium, large")
    parser.add_argument("--voz", default=OPENAI_TTS_VOICE,
                        help=f"Voz OpenAI: alloy, echo, fable, onyx, nova, shimmer (padrão: {OPENAI_TTS_VOICE})")
    parser.add_argument("--tts-modelo", default=OPENAI_TTS_MODEL,
                        choices=["tts-1", "tts-1-hd"],
                        help="Modelo TTS: tts-1 (rápido) ou tts-1-hd (qualidade máxima)")
    parser.add_argument("--agrupar", type=int, default=0, metavar="SEGUNDOS",
                        help="Agrupar segmentos em blocos de N segundos")
    parser.add_argument("--apenas-legendas", action="store_true",
                        help="Só gera legendas .srt, sem dublar")
    parser.add_argument("--manter-temp", action="store_true",
                        help="Mantém arquivos temporários")
    parser.add_argument("--sim", action="store_true",
                        help="Pular confirmação de custo estimado")

    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"❌ Arquivo não encontrado: {video_path}")
        sys.exit(1)

    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY não configurada!")
        print("   export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    verificar_dependencias()

    # Inicializar cliente OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Paths de output
    stem = video_path.stem
    output_video = video_path.parent / f"{stem}_{args.idioma.lower()}.mp4"
    output_srt = video_path.parent / f"{stem}_{args.idioma.lower()}.srt"

    temp_dir = Path(tempfile.mkdtemp(prefix="dublar2_"))

    print("=" * 60)
    print("🎬 DUBLAR 2.0 — Dublagem premium com IA")
    print("=" * 60)
    print(f"   Vídeo:      {video_path.name}")
    print(f"   Idioma:     {args.idioma}")
    print(f"   Whisper:    {args.modelo}")
    print(f"   Voz:        {args.voz} ({args.tts_modelo})")
    if args.agrupar > 0:
        print(f"   Agrupar:    blocos de {args.agrupar}s")
    print(f"   Temp:       {temp_dir}")
    print("=" * 60)

    try:
        # 1. Extrair áudio
        audio_path = extrair_audio(video_path, temp_dir)

        # 2. Transcrever
        segmentos = transcrever(audio_path, args.modelo)

        if not segmentos:
            print("\n❌ Nenhum áudio detectado no vídeo.")
            sys.exit(1)

        # 2.5. Agrupar (opcional)
        if args.agrupar > 0:
            segmentos = agrupar_segmentos(segmentos, args.agrupar)

        # Estimar custo
        custo = estimar_custo(segmentos)
        print(f"\n💰 Custo estimado:")
        print(f"   Tradução (GPT-4o-mini): ${custo['custo_traducao']:.4f}")
        print(f"   TTS (OpenAI):           ${custo['custo_tts']:.4f}")
        print(f"   Total:                  ${custo['custo_total']:.4f}")

        if not args.sim and not args.apenas_legendas:
            resposta = input("\n   Continuar? [S/n] ").strip().lower()
            if resposta in ("n", "no", "nao", "não"):
                print("   Cancelado.")
                sys.exit(0)

        # 3. Traduzir
        segmentos_traduzidos = traduzir(segmentos, client)

        # 4. Gerar legendas .srt
        gerar_srt(segmentos_traduzidos, output_srt)

        # 5. Gerar áudio e montar vídeo
        if not args.apenas_legendas:
            segmentos_com_audio = gerar_audio_tts(
                segmentos_traduzidos, temp_dir, client,
                args.voz, args.tts_modelo
            )

            duracao_video = obter_duracao_video(video_path)
            audio_final = montar_audio_final(
                segmentos_com_audio, temp_dir, duracao_video
            )
            montar_video_final(video_path, audio_final, output_video)

        # Resumo
        print("\n" + "=" * 60)
        print("✅ CONCLUÍDO!")
        print("=" * 60)
        if not args.apenas_legendas:
            print(f"   🎬 Vídeo dublado: {output_video}")
        print(f"   📄 Legendas:      {output_srt}")
        print(f"   💰 Custo real:    ~${custo['custo_total']:.4f}")
        print("=" * 60)

    finally:
        if not args.manter_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"\n   📁 Temp mantidos em: {temp_dir}")


if __name__ == "__main__":
    main()
