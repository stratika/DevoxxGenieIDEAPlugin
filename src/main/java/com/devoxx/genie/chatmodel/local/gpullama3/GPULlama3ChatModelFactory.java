package com.devoxx.genie.chatmodel.local.gpullama3;

import com.devoxx.genie.chatmodel.local.LocalChatModelFactory;
import com.devoxx.genie.model.CustomChatModel;
import com.devoxx.genie.model.LanguageModel;
import com.devoxx.genie.model.enumarations.ModelProvider;
import com.devoxx.genie.ui.settings.DevoxxGenieStateService;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/**
 * Factory for the out-of-process GPULlama3 Quarkus server. The server speaks the OpenAI
 * Chat Completions API, so we reuse {@link LocalChatModelFactory}'s OpenAI client and
 * just point it at the configured base URL. Model discovery hits {@code /v1/models}.
 *
 * @see <a href="https://github.com/devoxx/gpullama3-quarkus-server">gpullama3-quarkus-server</a>
 */
public class GPULlama3ChatModelFactory extends LocalChatModelFactory {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final int FETCH_TIMEOUT_SECONDS = 5;

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
    protected Object[] fetchModels() throws IOException {
        String baseUrl = getModelUrl();
        if (baseUrl == null || baseUrl.isBlank()) {
            throw new IOException("GPULlama3 server URL is not configured");
        }
        String modelsUrl = baseUrl.endsWith("/") ? baseUrl + "models" : baseUrl + "/models";
        try {
            HttpResponse<String> response = HttpClient.newBuilder()
                    .connectTimeout(Duration.ofSeconds(FETCH_TIMEOUT_SECONDS))
                    .build()
                    .send(HttpRequest.newBuilder(URI.create(modelsUrl))
                                    .timeout(Duration.ofSeconds(FETCH_TIMEOUT_SECONDS))
                                    .GET()
                                    .build(),
                            HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() != 200) {
                throw new IOException("GPULlama3 server returned HTTP " + response.statusCode());
            }
            JsonNode data = OBJECT_MAPPER.readTree(response.body()).get("data");
            if (data == null || !data.isArray()) {
                return new Object[0];
            }
            List<String> ids = new ArrayList<>();
            data.forEach(item -> {
                JsonNode id = item.get("id");
                if (id != null) ids.add(id.asText());
            });
            return ids.toArray();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while fetching GPULlama3 models", e);
        }
    }

    @Override
    protected LanguageModel buildLanguageModel(Object model) {
        String modelId = (String) model;
        return LanguageModel.builder()
                .provider(modelProvider)
                .modelName(modelId)
                .displayName(modelId)
                .inputCost(0)
                .outputCost(0)
                .inputMaxTokens(8192)
                .apiKeyUsed(false)
                .build();
    }
}
