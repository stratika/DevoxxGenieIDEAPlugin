package com.devoxx.genie.chatmodel.local.gpullama3;

import com.devoxx.genie.model.CustomChatModel;
import com.devoxx.genie.model.LanguageModel;
import com.devoxx.genie.model.enumarations.ModelProvider;
import com.devoxx.genie.model.gpullama3.GPULlama3ModelEntryDTO;
import com.devoxx.genie.ui.settings.DevoxxGenieStateService;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import org.junit.jupiter.api.Test;
import org.mockito.MockedStatic;
import org.mockito.Mockito;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class GPULlama3ChatModelFactoryTest {

    private static GPULlama3ModelEntryDTO modelEntry(String id) {
        GPULlama3ModelEntryDTO entry = new GPULlama3ModelEntryDTO();
        entry.setId(id);
        entry.setObject("model");
        entry.setOwnedBy("gpullama3");
        return entry;
    }

    @Test
    void createChatModel_usesConfiguredGpuLlama3Url() {
        try (MockedStatic<DevoxxGenieStateService> ignored = Mockito.mockStatic(DevoxxGenieStateService.class)) {
            DevoxxGenieStateService state = mock(DevoxxGenieStateService.class);
            when(DevoxxGenieStateService.getInstance()).thenReturn(state);
            when(state.getGpuLlama3ModelUrl()).thenReturn("http://localhost:8090/v1/");

            CustomChatModel customChatModel = new CustomChatModel();
            customChatModel.setModelName("Llama-3.2-1B-Instruct-Q8_0");

            ChatModel result = new GPULlama3ChatModelFactory().createChatModel(customChatModel);

            assertThat(result).isNotNull();
        }
    }

    @Test
    void createStreamingChatModel_usesConfiguredGpuLlama3Url() {
        try (MockedStatic<DevoxxGenieStateService> ignored = Mockito.mockStatic(DevoxxGenieStateService.class)) {
            DevoxxGenieStateService state = mock(DevoxxGenieStateService.class);
            when(DevoxxGenieStateService.getInstance()).thenReturn(state);
            when(state.getGpuLlama3ModelUrl()).thenReturn("http://localhost:8090/v1/");

            CustomChatModel customChatModel = new CustomChatModel();
            customChatModel.setModelName("Llama-3.2-1B-Instruct-Q8_0");

            StreamingChatModel result = new GPULlama3ChatModelFactory().createStreamingChatModel(customChatModel);

            assertThat(result).isNotNull();
        }
    }

    /**
     * GPULlama3 reports no context length on {@code /v1/models} (and {@code /health} carries only a
     * status), so an unconfigured fallback must land on the factory's documented default rather
     * than 0, which would break the token/usage bar.
     */
    @Test
    void buildLanguageModel_withoutConfiguredFallback_usesDefaultContextLength() {
        try (MockedStatic<DevoxxGenieStateService> ignored = Mockito.mockStatic(DevoxxGenieStateService.class)) {
            DevoxxGenieStateService state = mock(DevoxxGenieStateService.class);
            when(DevoxxGenieStateService.getInstance()).thenReturn(state);
            when(state.getGpuLlama3FallbackContextLength()).thenReturn(null);

            LanguageModel model = new GPULlama3ChatModelFactory()
                    .buildLanguageModel(modelEntry("Llama-3.2-1B-Instruct-Q8_0"));

            assertThat(model.getProvider()).isEqualTo(ModelProvider.GPULlama3);
            assertThat(model.getModelName()).isEqualTo("Llama-3.2-1B-Instruct-Q8_0");
            assertThat(model.getDisplayName()).isEqualTo("Llama-3.2-1B-Instruct-Q8_0");
            assertThat(model.getInputMaxTokens()).isEqualTo(GPULlama3ChatModelFactory.DEFAULT_CONTEXT_LENGTH);
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
            when(state.getGpuLlama3FallbackContextLength()).thenReturn(131_072);

            LanguageModel model = new GPULlama3ChatModelFactory()
                    .buildLanguageModel(modelEntry("Llama-3.2-1B-Instruct-Q8_0"));

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
            when(state.getGpuLlama3FallbackContextLength()).thenReturn(null);

            LanguageModel model = new GPULlama3ChatModelFactory()
                    .buildLanguageModel(new GPULlama3ModelEntryDTO());

            assertThat(model.getModelName()).isEmpty();
            assertThat(model.getDisplayName()).isEmpty();
        }
    }
}
