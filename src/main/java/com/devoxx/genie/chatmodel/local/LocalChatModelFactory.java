package com.devoxx.genie.chatmodel.local;

import com.devoxx.genie.chatmodel.ChatModelFactory;
import com.devoxx.genie.chatmodel.ThinkingSupport;
import com.devoxx.genie.model.CustomChatModel;
import com.devoxx.genie.model.LanguageModel;
import com.devoxx.genie.model.enumarations.ModelProvider;
import com.devoxx.genie.ui.util.NotificationUtil;
import com.intellij.openapi.project.ProjectManager;
import com.intellij.util.concurrency.AppExecutorUtil;

import dev.langchain4j.http.client.jdk.JdkHttpClient;
import dev.langchain4j.http.client.jdk.JdkHttpClientBuilder;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.net.http.HttpClient;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

public abstract class LocalChatModelFactory implements ChatModelFactory {

    protected final ModelProvider modelProvider;
    public List<LanguageModel> cachedModels = null;

    protected static boolean warningShown = false;
    public boolean providerRunning = false;
    public boolean providerChecked = false;

    /**
     * Why the last model probe failed, or {@code null} when it has not failed. Kept so the
     * notification can say what actually went wrong instead of always blaming the endpoint.
     */
    private volatile String lastProbeFailure = null;

    // LMStudio does not support HTTP_2, see https://github.com/langchain4j/langchain4j/issues/2758
    private final HttpClient.Builder httpClientBuilder = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_1_1) ;
    private final JdkHttpClientBuilder jdkHttpClientBuilder = JdkHttpClient.builder()
            .httpClientBuilder(httpClientBuilder);

    protected LocalChatModelFactory(ModelProvider modelProvider) {
        this.modelProvider = modelProvider;
    }

    @Override
    public abstract ChatModel createChatModel(@NotNull CustomChatModel customChatModel);

    @Override
    public abstract StreamingChatModel createStreamingChatModel(@NotNull CustomChatModel customChatModel);

    protected abstract String getModelUrl();

    /**
     * The langchain4j HTTP client builder used for the OpenAI-compatible chat models.
     * Exposed as a hook so providers with server-specific quirks can decorate it
     * (e.g. Jan compacts JSON request bodies, see issue #1051).
     */
    protected dev.langchain4j.http.client.HttpClientBuilder resolveHttpClientBuilder() {
        return jdkHttpClientBuilder;
    }

    protected ChatModel createOpenAiChatModel(@NotNull CustomChatModel customChatModel) {
        return OpenAiChatModel.builder()
                .baseUrl(getModelUrl())
                .httpClientBuilder(resolveHttpClientBuilder())
                .apiKey("na")
                .modelName(customChatModel.getModelName())
                .maxRetries(customChatModel.getMaxRetries())
                .temperature(customChatModel.getTemperature())
                .maxTokens(customChatModel.getMaxTokens())
                .timeout(Duration.ofSeconds(customChatModel.getTimeout()))
                .topP(customChatModel.getTopP())
                .returnThinking(ThinkingSupport.isEnabled())
                .listeners(getListener())
                .build();
    }

    protected StreamingChatModel createOpenAiStreamingChatModel(@NotNull CustomChatModel customChatModel) {
        return OpenAiStreamingChatModel.builder()
                .baseUrl(getModelUrl())
                .httpClientBuilder(resolveHttpClientBuilder())
                .apiKey("na")
                .modelName(customChatModel.getModelName())
                .temperature(customChatModel.getTemperature())
                .topP(customChatModel.getTopP())
                .maxTokens(customChatModel.getMaxTokens())
                .timeout(Duration.ofSeconds(customChatModel.getTimeout()))
                .returnThinking(ThinkingSupport.isEnabled())
                .listeners(getListener())
                .build();
    }

    @Override
    public List<LanguageModel> getModels() {
        if (!providerChecked) {
            checkAndFetchModels();
        }
        if (!providerRunning) {
            handleProviderNotRunning();
            return List.of();
        }
        return cachedModels;
    }

    protected void handleProviderNotRunning() {
        NotificationUtil.sendNotification(ProjectManager.getInstance().getDefaultProject(),
                lastProbeFailure != null
                        ? lastProbeFailure
                        : "LLM provider is not running. Please start it and try again.");
    }

    /**
     * What to tell the user when a model probe failed.
     *
     * <p>A probe fails for two unrelated reasons and they need different answers. An
     * {@link IOException} means the endpoint did not respond, so naming the URL lets the user
     * check the server. Anything else is a fault on this side of the wire — a misconfigured
     * factory, an unregistered service whose {@code @NotNull} accessor returned {@code null} —
     * and telling the user to start a provider that is already running sends them to the wrong
     * place entirely. That is not hypothetical: an unregistered {@code JitLLMModelService}
     * threw {@code IllegalStateException} out of {@code fetchModels}, escaped the
     * {@code IOException}-only catch, and left the provider reported as "not running" while the
     * server was answering {@code /v1/models} in under a millisecond.
     *
     * @param providerName the provider's display name
     * @param url the configured endpoint, or {@code null} when it is not known
     * @param cause what the probe threw
     */
    static String providerUnavailableMessage(String providerName, String url, Throwable cause) {
        String detail = cause.getMessage() == null ? cause.getClass().getSimpleName() : cause.getMessage();
        if (cause instanceof IOException) {
            return providerName + " did not respond at " + (url == null || url.isBlank() ? "its configured URL" : url)
                    + " (" + detail + "). Start it, or correct the URL in Settings.";
        }
        return providerName + " failed while listing models: " + cause.getClass().getSimpleName()
                + " (" + detail + "). This is a plugin-side error, not a stopped server — see the IDE log.";
    }

    private void checkAndFetchModels() {
        List<LanguageModel> modelNames = new ArrayList<>();
        List<CompletableFuture<Void>> futures = new ArrayList<>();
        try {
            Object[] models = fetchModels();
            for (Object model : models) {
                CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                    try {
                        LanguageModel languageModel = buildLanguageModel(model);
                        synchronized (modelNames) {
                            modelNames.add(languageModel);
                        }
                    } catch (IOException e) {
                        handleModelFetchError(e);
                    }
                }, AppExecutorUtil.getAppExecutorService());
                futures.add(future);
            }
            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
            cachedModels = modelNames;
            providerRunning = true;
            lastProbeFailure = null;
        } catch (IOException e) {
            handleGeneralFetchError(e);
            cachedModels = List.of();
            providerRunning = false;
            lastProbeFailure = providerUnavailableMessage(modelProvider.getName(), safeModelUrl(), e);
        } catch (RuntimeException e) {
            // Previously uncaught: a factory fault escaped here, so providerRunning stayed false
            // while providerChecked became true in the finally, and every later call reported the
            // provider as not running.
            cachedModels = List.of();
            providerRunning = false;
            lastProbeFailure = providerUnavailableMessage(modelProvider.getName(), safeModelUrl(), e);
        } finally {
            providerChecked = true;
        }
    }

    protected abstract Object[] fetchModels() throws IOException;

    protected abstract LanguageModel buildLanguageModel(Object model) throws IOException;

    protected void handleModelFetchError(@NotNull IOException e) {
        NotificationUtil.sendNotification(ProjectManager.getInstance().getDefaultProject(), "Error fetching model details: " + e.getMessage());
    }

    protected void handleGeneralFetchError(IOException e) {
        if (!warningShown) {
            NotificationUtil.sendNotification(ProjectManager.getInstance().getDefaultProject(), "Error fetching models: " + e.getMessage());
            warningShown = true;
        }
    }

    @Override
    public void resetModels() {
        cachedModels = null;
        providerChecked = false;
        providerRunning = false;
        lastProbeFailure = null;
    }

    /** The configured URL for the failure message; a broken {@code getModelUrl()} must not mask it. */
    private String safeModelUrl() {
        try {
            return getModelUrl();
        } catch (RuntimeException ignored) {
            return null;
        }
    }
}
