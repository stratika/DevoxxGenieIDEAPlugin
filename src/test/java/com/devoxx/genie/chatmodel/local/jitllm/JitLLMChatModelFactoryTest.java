package com.devoxx.genie.chatmodel.local.jitllm;

import com.devoxx.genie.model.CustomChatModel;
import com.devoxx.genie.model.LanguageModel;
import com.devoxx.genie.model.enumarations.ModelProvider;
import com.devoxx.genie.model.jitllm.JitLLMModelEntryDTO;
import com.devoxx.genie.model.jitllm.JitLLMModelsResponseDTO;
import com.google.gson.Gson;
import com.devoxx.genie.ui.settings.DevoxxGenieStateService;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import org.junit.jupiter.api.Test;
import org.mockito.MockedStatic;
import org.mockito.Mockito;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class JitLLMChatModelFactoryTest {

    private static JitLLMModelEntryDTO modelEntry(String id) {
        JitLLMModelEntryDTO entry = new JitLLMModelEntryDTO();
        entry.setId(id);
        entry.setObject("model");
        entry.setOwnedBy("jitllm");
        return entry;
    }

    @Test
    void createChatModel_usesConfiguredJitLLMUrl() {
        try (MockedStatic<DevoxxGenieStateService> ignored = Mockito.mockStatic(DevoxxGenieStateService.class)) {
            DevoxxGenieStateService state = mock(DevoxxGenieStateService.class);
            when(DevoxxGenieStateService.getInstance()).thenReturn(state);
            when(state.getJitLLMModelUrl()).thenReturn("http://localhost:8090/v1/");

            CustomChatModel customChatModel = new CustomChatModel();
            customChatModel.setModelName("Llama-3.2-1B-Instruct-Q8_0");

            ChatModel result = new JitLLMChatModelFactory().createChatModel(customChatModel);

            assertThat(result).isNotNull();
        }
    }

    @Test
    void createStreamingChatModel_usesConfiguredJitLLMUrl() {
        try (MockedStatic<DevoxxGenieStateService> ignored = Mockito.mockStatic(DevoxxGenieStateService.class)) {
            DevoxxGenieStateService state = mock(DevoxxGenieStateService.class);
            when(DevoxxGenieStateService.getInstance()).thenReturn(state);
            when(state.getJitLLMModelUrl()).thenReturn("http://localhost:8090/v1/");

            CustomChatModel customChatModel = new CustomChatModel();
            customChatModel.setModelName("Llama-3.2-1B-Instruct-Q8_0");

            StreamingChatModel result = new JitLLMChatModelFactory().createStreamingChatModel(customChatModel);

            assertThat(result).isNotNull();
        }
    }

    /**
     * jitLLM caps a reply at 256 tokens when the request carries no {@code max_tokens}, so the
     * streaming model must forward the configured limit just like the non-streaming one does —
     * otherwise every streamed chat reply is cut off mid-sentence.
     */
    @Test
    void createStreamingChatModel_forwardsMaxTokens() {
        try (MockedStatic<DevoxxGenieStateService> ignored = Mockito.mockStatic(DevoxxGenieStateService.class)) {
            DevoxxGenieStateService state = mock(DevoxxGenieStateService.class);
            when(DevoxxGenieStateService.getInstance()).thenReturn(state);
            when(state.getJitLLMModelUrl()).thenReturn("http://localhost:8090/v1/");

            CustomChatModel customChatModel = new CustomChatModel();
            customChatModel.setModelName("gemma-4-E2B-it-Q4_0");
            customChatModel.setMaxTokens(4000);

            StreamingChatModel result = new JitLLMChatModelFactory().createStreamingChatModel(customChatModel);

            assertThat(result.defaultRequestParameters().maxOutputTokens()).isEqualTo(4000);
        }
    }

    /**
     * Older jitLLM builds report no context length on {@code /v1/models}, so an unconfigured
     * fallback must land on the factory's documented default rather than 0, which would break the
     * token/usage bar.
     */
    @Test
    void buildLanguageModel_withoutConfiguredFallback_usesDefaultContextLength() {
        try (MockedStatic<DevoxxGenieStateService> ignored = Mockito.mockStatic(DevoxxGenieStateService.class)) {
            DevoxxGenieStateService state = mock(DevoxxGenieStateService.class);
            when(DevoxxGenieStateService.getInstance()).thenReturn(state);
            when(state.getJitLLMFallbackContextLength()).thenReturn(null);

            LanguageModel model = new JitLLMChatModelFactory()
                    .buildLanguageModel(modelEntry("Llama-3.2-1B-Instruct-Q8_0"));

            assertThat(model.getProvider()).isEqualTo(ModelProvider.JitLLM);
            assertThat(model.getModelName()).isEqualTo("Llama-3.2-1B-Instruct-Q8_0");
            assertThat(model.getDisplayName()).isEqualTo("Llama-3.2-1B-Instruct-Q8_0");
            assertThat(model.getInputMaxTokens()).isEqualTo(JitLLMChatModelFactory.DEFAULT_CONTEXT_LENGTH);
            assertThat(model.isApiKeyUsed()).isFalse();
            assertThat(model.getInputCost()).isZero();
            assertThat(model.getOutputCost()).isZero();
        }
    }

    @Test
    void buildLanguageModel_withConfiguredFallback_usesConfiguredContextLength() {
        try (MockedStatic<DevoxxGenieStateService> ignored = Mockito.mockStatic(DevoxxGenieStateService.class)) {
            DevoxxGenieStateService state = mock(DevoxxGenieStateService.class);
            when(DevoxxGenieStateService.getInstance()).thenReturn(state);
            when(state.getJitLLMFallbackContextLength()).thenReturn(131_072);

            LanguageModel model = new JitLLMChatModelFactory()
                    .buildLanguageModel(modelEntry("Llama-3.2-1B-Instruct-Q8_0"));

            assertThat(model.getInputMaxTokens()).isEqualTo(131_072);
        }
    }

    /**
     * Newer jitLLM builds report the {@code --ctx-size} the server was started with as
     * {@code context_length}. That is the real window, so it wins over the user's fallback.
     */
    @Test
    void buildLanguageModel_withServerReportedContextLength_usesReportedValue() {
        String body = "{\"object\":\"list\",\"data\":[{\"id\":\"gemma-4-E2B-it-Q4_0\",\"object\":\"model\","
                + "\"created\":0,\"owned_by\":\"jitllm\",\"context_length\":131072}]}";
        JitLLMModelEntryDTO entry = new Gson().fromJson(body, JitLLMModelsResponseDTO.class).getData().get(0);

        try (MockedStatic<DevoxxGenieStateService> ignored = Mockito.mockStatic(DevoxxGenieStateService.class)) {
            DevoxxGenieStateService state = mock(DevoxxGenieStateService.class);
            when(DevoxxGenieStateService.getInstance()).thenReturn(state);
            when(state.getJitLLMFallbackContextLength()).thenReturn(8_000);

            LanguageModel model = new JitLLMChatModelFactory().buildLanguageModel(entry);

            assertThat(model.getInputMaxTokens()).isEqualTo(131_072);
        }
    }

    /**
     * The served id comes straight off the GGUF file name upstream, but a malformed or empty
     * {@code /v1/models} entry must not produce a null model name in the dropdown.
     */
    @Test
    void buildLanguageModel_withNullId_yieldsEmptyModelName() {
        try (MockedStatic<DevoxxGenieStateService> ignored = Mockito.mockStatic(DevoxxGenieStateService.class)) {
            DevoxxGenieStateService state = mock(DevoxxGenieStateService.class);
            when(DevoxxGenieStateService.getInstance()).thenReturn(state);
            when(state.getJitLLMFallbackContextLength()).thenReturn(null);

            LanguageModel model = new JitLLMChatModelFactory()
                    .buildLanguageModel(new JitLLMModelEntryDTO());

            assertThat(model.getModelName()).isEmpty();
            assertThat(model.getDisplayName()).isEmpty();
        }
    }
}
