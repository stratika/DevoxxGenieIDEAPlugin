package com.devoxx.genie.chatmodel.local.jitllm;

import com.devoxx.genie.model.jitllm.JitLLMModelsResponseDTO;
import com.google.gson.Gson;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class JitLLMModelServiceTest {

    private static class TestableModelService extends JitLLMModelService {
        String testBuildModelsUrl(String baseUrl) {
            return buildModelsUrl(baseUrl);
        }
    }

    private final TestableModelService service = new TestableModelService();
    private final Gson gson = new Gson();

    @Test
    void buildModelsUrl_withTrailingSlash_appendsModels() {
        assertThat(service.testBuildModelsUrl("http://localhost:8090/v1/"))
                .isEqualTo("http://localhost:8090/v1/models");
    }

    @Test
    void buildModelsUrl_withoutTrailingSlash_insertsSeparator() {
        assertThat(service.testBuildModelsUrl("http://localhost:8090/v1"))
                .isEqualTo("http://localhost:8090/v1/models");
    }

    /** jitLLM's own default is 8080; users who don't pass --port 8090 just edit the URL. */
    @Test
    void buildModelsUrl_withUpstreamDefaultPort_preservesHostAndPort() {
        assertThat(service.testBuildModelsUrl("http://127.0.0.1:8080/v1/"))
                .isEqualTo("http://127.0.0.1:8080/v1/models");
    }

    @Test
    void buildModelsUrl_withNullBaseUrl_returnsDefaultUrl() {
        assertThat(service.testBuildModelsUrl(null)).isEqualTo(JitLLMModelService.DEFAULT_MODELS_URL);
    }

    @Test
    void buildModelsUrl_withBlankBaseUrl_returnsDefaultUrl() {
        assertThat(service.testBuildModelsUrl("   ")).isEqualTo(JitLLMModelService.DEFAULT_MODELS_URL);
    }

    @Test
    void buildModelsUrl_withSurroundingWhitespace_trimsBeforeAppending() {
        assertThat(service.testBuildModelsUrl("  http://localhost:8090/v1/  "))
                .isEqualTo("http://localhost:8090/v1/models");
    }

    /**
     * Guards the Gson binding against the exact payload jitLLM's {@code handleModels} emits:
     * a single-entry list whose id is the GGUF file name with the suffix stripped.
     */
    @Test
    void responseDto_parsesJitLLMModelsPayload() {
        String json = """
                {
                  "object": "list",
                  "data": [
                    {"id": "Llama-3.2-1B-Instruct-Q8_0", "object": "model", "created": 0, "owned_by": "jitllm"}
                  ]
                }
                """;

        JitLLMModelsResponseDTO response = gson.fromJson(json, JitLLMModelsResponseDTO.class);

        assertThat(response.getObject()).isEqualTo("list");
        assertThat(response.getData()).hasSize(1);
        assertThat(response.getData().get(0).getId()).isEqualTo("Llama-3.2-1B-Instruct-Q8_0");
        assertThat(response.getData().get(0).getObject()).isEqualTo("model");
        assertThat(response.getData().get(0).getCreated()).isZero();
        assertThat(response.getData().get(0).getOwnedBy()).isEqualTo("jitllm");
    }

    @Test
    void responseDto_withEmptyModelList_yieldsEmptyData() {
        JitLLMModelsResponseDTO response =
                gson.fromJson("{\"object\":\"list\",\"data\":[]}", JitLLMModelsResponseDTO.class);

        assertThat(response.getData()).isEmpty();
    }
}
