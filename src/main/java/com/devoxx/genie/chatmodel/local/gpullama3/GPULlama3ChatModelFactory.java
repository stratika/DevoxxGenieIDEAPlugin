package com.devoxx.genie.chatmodel.local.gpullama3;

import com.devoxx.genie.chatmodel.local.LocalChatModelFactory;
import com.devoxx.genie.model.CustomChatModel;
import com.devoxx.genie.model.LanguageModel;
import com.devoxx.genie.model.enumarations.ModelProvider;
import com.devoxx.genie.model.gpullama3.GPULlama3ModelEntryDTO;
import com.devoxx.genie.ui.settings.DevoxxGenieStateService;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;

/**
 * GPULlama3 (<a href="https://github.com/beehive-lab/GPULlama3.java">beehive-lab/GPULlama3.java</a>)
 * runs GGUF models on the GPU through TornadoVM. Since v1.0.0 it serves them over its own
 * OpenAI-compatible HTTP server ({@code llama-tornado --server}), so it plugs straight into the
 * shared OpenAI chat/streaming clients — no intermediate bridge service is required.
 */
public class GPULlama3ChatModelFactory extends LocalChatModelFactory {

    /**
     * GPULlama3's {@code /v1/models} response carries no context-length field (see
     * {@link GPULlama3ModelEntryDTO}) and {@code /health} reports only liveness, so the window has
     * to be assumed. 8k is the conservative floor that the Llama-3 family meets; users running
     * larger-context models raise it via the "GPULlama3 Fallback Context" setting.
     */
    public static final int DEFAULT_CONTEXT_LENGTH = 8000;

    public GPULlama3ChatModelFactory() {
        super(ModelProvider.GPULlama3);
    }

    @Override
    public ChatModel createChatModel(@NotNull CustomChatModel customChatModel) {
        return createOpenAiChatModel(customChatModel);
    }

    @Override
    public StreamingChatModel createStreamingChatModel(@NotNull CustomChatModel customChatModel) {
        return createOpenAiStreamingChatModel(customChatModel);
    }

    @Override
    protected String getModelUrl() {
        return DevoxxGenieStateService.getInstance().getGpuLlama3ModelUrl();
    }

    @Override
    protected GPULlama3ModelEntryDTO[] fetchModels() throws IOException {
        return GPULlama3ModelService.getInstance().getModels().toArray(new GPULlama3ModelEntryDTO[0]);
    }

    @Override
    protected LanguageModel buildLanguageModel(Object model) {
        GPULlama3ModelEntryDTO gpuLlama3Model = (GPULlama3ModelEntryDTO) model;
        Integer configuredFallback = DevoxxGenieStateService.getInstance().getGpuLlama3FallbackContextLength();
        String modelId = gpuLlama3Model.getId() == null ? "" : gpuLlama3Model.getId();
        return LanguageModel.builder()
                .provider(modelProvider)
                .modelName(modelId)
                .displayName(modelId)
                .inputCost(0)
                .outputCost(0)
                .inputMaxTokens(configuredFallback != null ? configuredFallback : DEFAULT_CONTEXT_LENGTH)
                .apiKeyUsed(false)
                .build();
    }
}
