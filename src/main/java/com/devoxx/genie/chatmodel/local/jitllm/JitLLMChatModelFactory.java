package com.devoxx.genie.chatmodel.local.jitllm;

import com.devoxx.genie.chatmodel.local.LocalChatModelFactory;
import com.devoxx.genie.model.CustomChatModel;
import com.devoxx.genie.model.LanguageModel;
import com.devoxx.genie.model.enumarations.ModelProvider;
import com.devoxx.genie.model.jitllm.JitLLMModelEntryDTO;
import com.devoxx.genie.ui.settings.DevoxxGenieStateService;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;

/**
 * jitLLM (<a href="https://github.com/beehive-lab/jitllm">beehive-lab/jitllm</a>)
 * runs GGUF models on the GPU through TornadoVM. Since v1.0.0 it serves them over its own
 * OpenAI-compatible HTTP server ({@code jitllm serve}), so it plugs straight into the
 * shared OpenAI chat/streaming clients — no intermediate bridge service is required.
 */
public class JitLLMChatModelFactory extends LocalChatModelFactory {

    /**
     * jitLLM's {@code /v1/models} response carries no context-length field (see
     * {@link JitLLMModelEntryDTO}) and {@code /health} reports only liveness, so the window has
     * to be assumed. 8k is the conservative floor that the Llama-3 family meets; users running
     * larger-context models raise it via the "jitLLM Fallback Context" setting.
     */
    public static final int DEFAULT_CONTEXT_LENGTH = 8000;

    public JitLLMChatModelFactory() {
        super(ModelProvider.JitLLM);
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
        return DevoxxGenieStateService.getInstance().getJitLLMModelUrl();
    }

    @Override
    protected JitLLMModelEntryDTO[] fetchModels() throws IOException {
        return JitLLMModelService.getInstance().getModels().toArray(new JitLLMModelEntryDTO[0]);
    }

    @Override
    protected LanguageModel buildLanguageModel(Object model) {
        JitLLMModelEntryDTO jitLLMModel = (JitLLMModelEntryDTO) model;
        Integer configuredFallback = DevoxxGenieStateService.getInstance().getJitLLMFallbackContextLength();
        String modelId = jitLLMModel.getId() == null ? "" : jitLLMModel.getId();
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
