# Running DevoxxGenie with GPULlama3

[GPULlama3.java](https://github.com/beehive-lab/GPULlama3.java) runs GGUF models on the GPU
through TornadoVM. Since **v1.0.0** it ships its own OpenAI-compatible HTTP server, so
DevoxxGenie talks to it directly with the OpenAI client it already uses for every other
local provider:

```
DevoxxGenie plugin  ──HTTP (OpenAI API)──►  GPULlama3 server  ──►  TornadoVM
```

No bridge or adapter process is involved. The plugin holds no GPULlama3, TornadoVM or Quarkus
dependency — it only needs a URL that speaks the OpenAI API.

> **Earlier versions.** GPULlama3 gained `/v1/chat/completions` in v1.0.0 (`OpenAIServer`).
> On v0.x there is no server to point at, so this integration requires v1.0.0 or newer.

---

## 1. Start the GPULlama3 server

```bash
llama-tornado --server --model /path/to/Llama-3.2-1B-Instruct-Q8_0.gguf --port 8090 --gpu
```

| Flag | Notes |
|------|-------|
| `--server` | Starts the OpenAI-compatible HTTP server. |
| `--model` | The GGUF file to serve. One model per server process. |
| `--port` | GPULlama3 defaults to **8080**; DevoxxGenie defaults to **8090** (see below). |
| `--gpu` | Use TornadoVM. Omit it to run on CPU, which is the safer first test. |

The server exposes:

- `POST /v1/chat/completions` — streaming (SSE) and non-streaming
- `POST /v1/completions`
- `GET /v1/models` — the single served model
- `GET /health` — liveness

### Why port 8090

DevoxxGenie already defaults Llama.c++ **and** Nativ to port 8080. Defaulting GPULlama3 there
too would make three providers collide, so the plugin ships with `http://localhost:8090/v1/`
and the command above passes `--port 8090` to match. If you prefer the upstream default, start
the server without `--port` and change the URL in settings to `http://localhost:8080/v1/`.

### Smoke-test before involving the plugin

```bash
curl -s http://localhost:8090/v1/models

curl -s http://localhost:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Llama-3.2-1B-Instruct-Q8_0","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

If both return JSON, the server is healthy end-to-end.

---

## 2. Point DevoxxGenie at it

1. **Settings → DevoxxGenie → LLM Providers → Local LLM Providers**
2. Tick **GPULlama3** (it ships disabled, since it needs a server started by hand).
3. Confirm the URL is `http://localhost:8090/v1/`.
4. Optionally set **GPULlama3 Fallback Context** — see below.
5. Apply, then pick **GPULlama3** in the provider dropdown.

The model dropdown is populated from `GET /v1/models`, so it shows the one model the server was
started with, named after the GGUF file without its suffix
(e.g. `Llama-3.2-1B-Instruct-Q8_0`).

### Context window

GPULlama3 reports no context length anywhere over HTTP — `/v1/models` carries only
`id`/`object`/`created`/`owned_by`, and `/health` returns just a status. DevoxxGenie therefore
assumes **8000 tokens** (`GPULlama3ChatModelFactory.DEFAULT_CONTEXT_LENGTH`), which is the
conservative floor for the Llama-3 family.

If you serve a larger-context model, enable **GPULlama3 Fallback Context** and set the real
value. It only drives the token-usage bar and the "context exceeded" warning; requests are sent
either way, so a wrong value shows misleading numbers rather than breaking chat.

### Generation parameters

Unlike a v0.x setup, the server honours per-request `temperature`, `top_p`, `max_tokens` and
`seed`, so the values in **LLM Settings** take effect.

---

## Configuration reference

| Setting | Default | Where |
|---------|---------|-------|
| GPULlama3 enabled | off | Settings → LLM Providers |
| GPULlama3 URL | `http://localhost:8090/v1/` | Settings → LLM Providers |
| GPULlama3 Fallback Context | unset → 8000 tokens | Settings → LLM Providers |
| Request timeout | `500` s | `Constant.TIMEOUT`; configurable in settings |

---

## Troubleshooting

### `ConnectException` / no models in the dropdown
The server is not running or the port does not match. Confirm with
`curl -s http://localhost:8090/v1/models`.

### `HttpTimeoutException: request timed out`
The connection succeeded but generation did not finish in time. On CPU, a large `max_tokens`
is slow — lower it, shorten the prompt, or raise the timeout in settings.

### The token-usage bar turns red on modest prompts
The assumed 8000-token window is too small for your model. Set **GPULlama3 Fallback Context**
to the model's real context length.

### Native crash in GPU mode
`malloc: *** error for object ...: pointer being freed was not allocated` and similar native
faults come from the TornadoVM/Metal layer, upstream of DevoxxGenie. Re-run without `--gpu` to
confirm, and report it to
[beehive-lab/GPULlama3.java](https://github.com/beehive-lab/GPULlama3.java).

---

## Implementation notes

- `GPULlama3ChatModelFactory` extends `LocalChatModelFactory`, reusing the shared OpenAI
  chat/streaming clients; it only overrides the base URL, model discovery and the context window.
- `GPULlama3ModelService` appends `models` to the configured base URL, tolerating a missing
  trailing slash.
- GPULlama3 serializes inference on one GPU context, so concurrent prompts queue server-side.
