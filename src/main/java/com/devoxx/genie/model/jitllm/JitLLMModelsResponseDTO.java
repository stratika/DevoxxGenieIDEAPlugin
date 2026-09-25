package com.devoxx.genie.model.jitllm;

import com.google.gson.annotations.SerializedName;
import lombok.Getter;
import lombok.Setter;

import java.util.List;

/**
 * Envelope of jitLLM's OpenAI-compatible {@code GET /v1/models} response:
 * {@code {"object": "list", "data": [ ... ]}}.
 */
@Getter
@Setter
public class JitLLMModelsResponseDTO {

    @SerializedName("object")
    private String object;

    @SerializedName("data")
    private List<JitLLMModelEntryDTO> data;
}
