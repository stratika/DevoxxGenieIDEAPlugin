package com.devoxx.genie.chatmodel.local.gpullama3;

import com.devoxx.genie.chatmodel.local.LocalLLMProvider;
import com.devoxx.genie.chatmodel.local.LocalLLMProviderUtil;
import com.devoxx.genie.model.gpullama3.GPULlama3ModelEntryDTO;
import com.devoxx.genie.model.gpullama3.GPULlama3ModelsResponseDTO;
import com.devoxx.genie.ui.settings.DevoxxGenieStateService;
import com.intellij.openapi.application.ApplicationManager;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.util.List;

/**
 * Lists the model GPULlama3 is currently serving.
 * <p>
 * Since v1.0.0 GPULlama3 ships its own OpenAI-compatible HTTP server
 * ({@code llama-tornado --server}), so the configured chat base URL
 * ({@code http://localhost:8090/v1/} by default) already points at the right place and only the
 * {@code models} segment has to be appended — no separate bridge process is involved.
 * <p>
 * The response always holds exactly one entry: the server loads a single GGUF model at startup.
 */
public class GPULlama3ModelService implements LocalLLMProvider {

    static final String DEFAULT_MODELS_URL = "http://localhost:8090/v1/models";

    @NotNull
    public static GPULlama3ModelService getInstance() {
        return ApplicationManager.getApplication().getService(GPULlama3ModelService.class);
    }

    @Override
    public List<GPULlama3ModelEntryDTO> getModels() throws IOException {
        GPULlama3ModelsResponseDTO response = LocalLLMProviderUtil
                .getModelsFromUrl(buildModelsUrl(DevoxxGenieStateService.getInstance().getGpuLlama3ModelUrl()),
                        GPULlama3ModelsResponseDTO.class);

        return response == null || response.getData() == null ? List.of() : response.getData();
    }

    /**
     * Appends {@code models} to the configured base URL, tolerating a missing trailing slash.
     *
     * @param baseUrl the configured GPULlama3 base URL, e.g. {@code http://localhost:8090/v1/}
     * @return the full models endpoint, e.g. {@code http://localhost:8090/v1/models}
     */
    protected String buildModelsUrl(String baseUrl) {
        if (baseUrl == null || baseUrl.isBlank()) {
            return DEFAULT_MODELS_URL;
        }
        String trimmed = baseUrl.trim();
        return trimmed.endsWith("/") ? trimmed + "models" : trimmed + "/models";
    }
}
