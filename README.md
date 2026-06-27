# Dublar 2.0 — Dublagem premium de vídeos com IA

Pipeline de dublagem de alta qualidade usando modelos de última geração:
- **Whisper** (OpenAI) — transcrição precisa do áudio
- **GPT-4o-mini** — tradução contextual especializada em programação
- **OpenAI TTS** — vozes naturais de alta qualidade

## Diferenças para a V1

| Aspecto | V1 (dublar/) | V2 (dublar-2.0/) |
|---------|-------------|------------------|
| Tradução | DeepL (genérica) | GPT-4o-mini (contextual, entende código) |
| Voz | Edge TTS (sintética) | OpenAI TTS (quase humana) |
| Termos técnicos | Lista fixa de proteção | GPT sabe quais termos manter em inglês |
| Custo | $0/mês | ~$0.35/vídeo de 30 min |
| Qualidade | Boa | Excelente |

## Custo estimado

| Componente | Por vídeo de 30 min | Por 10 vídeos/mês |
|---|---|---|
| Whisper (local) | $0 | $0 |
| GPT-4o-mini (tradução) | ~$0.05 | ~$0.50 |
| OpenAI TTS (voz) | ~$0.30 | ~$3.00 |
| **Total** | **~$0.35** | **~$3.50** |

---

## Setup (Mac M4 — Apple Silicon)

### 1. Homebrew + FFmpeg

```bash
brew install ffmpeg
```

### 2. Python + Virtual Environment

```bash
cd tools/dublar-2.0

# Criar venv
python3 -m venv venv

# Ativar
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 3. API Key da OpenAI

1. Crie conta em https://platform.openai.com
2. Gere uma API key em https://platform.openai.com/api-keys
3. Adicione créditos (mínimo $5 — dura ~14 vídeos de 30 min)
4. Configure:

```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.zshrc
source ~/.zshrc
```

### 4. Verificar instalação

```bash
source venv/bin/activate
python3 dublar.py --help
```

---

## Uso

```bash
# Ativar venv (toda vez que abrir terminal novo)
cd tools/dublar-2.0
source venv/bin/activate

# Dublagem padrão
python3 dublar.py ~/Downloads/aula01.mov

# Recomendado para cursos de programação
python3 dublar.py ~/Downloads/aula01.mov --modelo small --agrupar 30

# Qualidade máxima (voz HD, custa o dobro em TTS)
python3 dublar.py ~/Downloads/aula01.mov --modelo small --agrupar 30 --tts-modelo tts-1-hd

# Só legendas (sem gerar áudio — mais barato)
python3 dublar.py ~/Downloads/aula01.mov --modelo small --apenas-legendas

# Pular confirmação de custo
python3 dublar.py ~/Downloads/aula01.mov --modelo small --agrupar 30 --sim
```

---

## Opções

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--modelo` | `base` | Modelo Whisper: tiny, base, small, medium, large |
| `--voz` | `onyx` | Voz OpenAI: alloy, echo, fable, onyx, nova, shimmer |
| `--tts-modelo` | `tts-1` | tts-1 (rápido, $15/1M chars) ou tts-1-hd (HD, $30/1M chars) |
| `--agrupar` | desativado | Agrupar segmentos em blocos de N segundos |
| `--apenas-legendas` | - | Só gera .srt |
| `--manter-temp` | - | Mantém arquivos temporários |
| `--sim` | - | Pula confirmação de custo |

### Vozes OpenAI

| Voz | Descrição |
|-----|-----------|
| `onyx` | Masculina, grave, profissional (padrão) |
| `nova` | Feminina, clara, amigável |
| `alloy` | Neutra, equilibrada |
| `echo` | Masculina, suave |
| `fable` | Narradora, expressiva |
| `shimmer` | Feminina, calorosa |

Recomendo testar `onyx` e `nova` pra ver qual prefere para estudo.

### Modelos Whisper

| Modelo | Velocidade | Qualidade | Recomendação |
|--------|-----------|-----------|--------------|
| base | Rápido | Boa | Testes rápidos |
| small | Médio | Muito boa | **Uso diário** |
| medium | Lento | Excelente | Áudio difícil |

### Agrupamento (`--agrupar`)

```bash
# Blocos de 30s (recomendado)
python3 dublar.py video.mov --agrupar 30

# Blocos de 60s (máxima fluidez)
python3 dublar.py video.mov --agrupar 60
```

Benefícios:
- Voz mais fluida (TTS gera entonação melhor com textos longos)
- Tradução mais coerente (GPT recebe mais contexto)
- Sincronização ajustada por bloco

---

## Output

Para `aula01.mov`:
- `aula01_pt-br.mp4` — vídeo dublado
- `aula01_pt-br.srt` — legendas em português

---

## Confirmação de custo

Antes de iniciar tradução e TTS, o script mostra o custo estimado e pede confirmação:

```
💰 Custo estimado:
   Tradução (GPT-4o-mini): $0.0480
   TTS (OpenAI):           $0.2880
   Total:                  $0.3360

   Continuar? [S/n]
```

Use `--sim` para pular essa confirmação em scripts automatizados.

---

## Formatos suportados

Entrada: `.mov`, `.mp4`, `.mkv`, `.avi`, `.webm`
Saída: `.mp4` (H.264 + AAC)

---

## Dica: comando completo recomendado para cursos

```bash
python3 dublar.py ~/Downloads/aula.mov --modelo small --agrupar 30 --voz onyx --sim
```
