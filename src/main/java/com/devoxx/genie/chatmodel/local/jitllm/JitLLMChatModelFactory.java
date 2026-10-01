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
     * Used only when the server does not report {@code context_length} on {@code /v1/models}
     * (older jitLLM builds, see {@link JitLLMModelEntryDTO}) and no "jitLLM Fallback Context" is
     * configured. 8k is the conservative floor that the Llama-3 family meets.
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
        String modelId = jitLLMModel.getId() == null ? "" : jitLLMModel.getId();
        return LanguageModel.builder()
                .provider(modelProvider)
                .modelName(modelId)
                .displayName(modelId)
                .inputCost(0)
                .outputCost(0)
                .inputMaxTokens(resolveContextLength(jitLLMModel))
                .apiKeyUsed(false)
                .build();
    }

    private static int resolveContextLength(@NotNull JitLLMModelEntryDTO model) {
        Integer reported = model.getContextLength();
        if (reported != null && reported > 0) {
            return reported;
        }
        Integer configuredFallback = DevoxxGenieStateService.getInstance().getJitLLMFallbackContextLength();
        return configuredFallback != null ? configuredFallback : DEFAULT_CONTEXT_LENGTH;
    }
}
