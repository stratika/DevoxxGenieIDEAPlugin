package com.devoxx.genie.model.gpullama3;

import com.google.gson.annotations.SerializedName;
import lombok.Getter;
import lombok.Setter;

import java.util.List;

/**
 * Envelope of GPULlama3's OpenAI-compatible {@code GET /v1/models} response:
 * {@code {"object": "list", "data": [ ... ]}}.
 */
@Getter
@Setter
public class GPULlama3ModelsResponseDTO {

    @SerializedName("object")
    private String object;

    @SerializedName("data")
    private List<GPULlama3ModelEntryDTO> data;
}
