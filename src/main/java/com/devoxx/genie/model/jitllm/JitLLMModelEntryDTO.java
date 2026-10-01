package com.devoxx.genie.model.jitllm;

import com.google.gson.annotations.SerializedName;
import lombok.Getter;
import lombok.Setter;

/**
 * One entry of jitLLM's OpenAI-compatible {@code GET /v1/models} response.
 * <p>
 * The server serves exactly one model — the GGUF file it was started with — and reports it with
 * the bare OpenAI model shape. The id is already the file name with the {@code .gguf} suffix
 * stripped (e.g. {@code Llama-3.2-1B-Instruct-Q8_0}), so it is usable as a display name as-is.
 * <p>
 * Newer builds also report {@code context_length} (the server's {@code --ctx-size}). Older ones
 * omit it, leaving the field null, in which case {@code JitLLMChatModelFactory} falls back to a
 * user-configurable value.
 */
@Getter
@Setter
public class JitLLMModelEntryDTO {

    @SerializedName("id")
    private String id;

    @SerializedName("object")
    private String object;

    @SerializedName("created")
    private Long created;

    @SerializedName("owned_by")
    private String ownedBy;

    @SerializedName("context_length")
    private Integer contextLength;
}
