package com.devoxx.genie.chatmodel.local.gpullama3;

import com.devoxx.genie.chatmodel.ChatModelFactory;
import com.devoxx.genie.model.CustomChatModel;
import com.devoxx.genie.model.LanguageModel;
import com.devoxx.genie.model.enumarations.ModelProvider;
import com.devoxx.genie.ui.settings.DevoxxGenieStateService;
import com.devoxx.genie.ui.util.NotificationUtil;
import com.intellij.openapi.project.ProjectManager;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.lang.reflect.Method;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Factory for beehive-lab/GPULlama3.java via dev.langchain4j:langchain4j-gpu-llama3.
 * <p>
 * Runs in-process — no HTTP, no port. The user supplies a path to a local .gguf file
 * and optionally enables GPU mode (which requires a TornadoVM-enabled JBR).
 * <p>
 * Both the chat model classes and their TornadoVM dependencies target JDK 25, so this
 * factory loads them via reflection. On older JBRs (e.g. the JBR 21 IntelliJ ships by
 * default) class-loading throws {@link UnsupportedClassVersionError} or
 * {@link NoClassDefFoundError}; we catch those and surface a friendly notification.
 *
 * @see <a href="https://github.com/beehive-lab/GPULlama3.java">beehive-lab/GPULlama3.java</a>
 */
public class GPULlama3ChatModelFactory implements ChatModelFactory {

    private static final String CHAT_MODEL_CLASS = "dev.langchain4j.model.gpullama3.GPULlama3ChatModel";
    private static final String STREAMING_CHAT_MODEL_CLASS = "dev.langchain4j.model.gpullama3.GPULlama3StreamingChatModel";

    @Override
    public ChatModel createChatModel(@NotNull CustomChatModel customChatModel) {
        Path modelPath = resolveModelPathOrFail();
        return (ChatModel) buildModel(CHAT_MODEL_CLASS, modelPath, customChatModel);
    }

    @Override
    public StreamingChatModel createStreamingChatModel(@NotNull CustomChatModel customChatModel) {
        Path modelPath = resolveModelPathOrFail();
        return (StreamingChatModel) buildModel(STREAMING_CHAT_MODEL_CLASS, modelPath, customChatModel);
    }

    @Override
    public List<LanguageModel> getModels() {
        String path = DevoxxGenieStateService.getInstance().getGpuLlama3ModelPath();
        if (path == null || path.isBlank()) {
            return List.of();
        }
        String displayName = Paths.get(path).getFileName().toString();
        LanguageModel model = LanguageModel.builder()
                .provider(ModelProvider.GPULlama3)
                .modelName(displayName)
                .displayName(displayName)
                .inputCost(0)
                .outputCost(0)
                .inputMaxTokens(8192)
                .apiKeyUsed(false)
                .build();
        List<LanguageModel> models = new ArrayList<>();
        models.add(model);
        return models;
    }

    private Path resolveModelPathOrFail() {
        String path = DevoxxGenieStateService.getInstance().getGpuLlama3ModelPath();
        if (path == null || path.isBlank()) {
            throw notify("GPULlama3 model path is not configured. Set it in Settings → Large Language Models.", null);
        }
        Path resolved = Paths.get(path);
        if (!java.nio.file.Files.exists(resolved)) {
            throw notify("GPULlama3 model file does not exist: " + path, null);
        }
        return resolved;
    }

    private @NotNull Object buildModel(@NotNull String modelClassName,
                                       @NotNull Path modelPath,
                                       @NotNull CustomChatModel customChatModel) {
        try {
            Class<?> modelClass = Class.forName(modelClassName);
            Method builderMethod = modelClass.getMethod("builder");
            Object builder = builderMethod.invoke(null);

            boolean useGpu = DevoxxGenieStateService.getInstance().isGpuLlama3UseGpu();

            invokeBuilder(builder, "modelPath", Path.class, modelPath);
            invokeBuilder(builder, "temperature", Double.class, customChatModel.getTemperature());
            invokeBuilder(builder, "topP", Double.class, customChatModel.getTopP());
            invokeBuilder(builder, "maxTokens", Integer.class, customChatModel.getMaxTokens());
            invokeBuilder(builder, "onGPU", Boolean.class, useGpu);

            Method buildMethod = builder.getClass().getMethod("build");
            return buildMethod.invoke(builder);
        } catch (Throwable t) {
            // IntelliJ's PluginClassLoader wraps loading failures in PluginException,
            // so we walk the cause chain rather than relying on individual catches.
            Throwable root = root(t);
            if (root instanceof UnsupportedClassVersionError) {
                String msg = root.getMessage() == null ? "" : root.getMessage();
                if (msg.contains("Preview features are not enabled")) {
                    throw notify("GPULlama3 requires JVM '--enable-preview'. Add it via Help → Edit Custom VM Options…, then restart the IDE.", t);
                }
                throw notify("GPULlama3 requires a newer JBR. Switch the IDE runtime via 'Choose Boot Java Runtime for the IDE…'.", t);
            }
            if (root instanceof ClassNotFoundException) {
                String missing = root.getMessage() == null ? "" : root.getMessage();
                if (missing.contains("jdk.incubator.vector") || missing.contains("jdk/incubator/vector")) {
                    throw notify("GPULlama3 needs the Vector API. Add '--add-modules=jdk.incubator.vector' via Help → Edit Custom VM Options…, then restart the IDE.", t);
                }
                throw notify("langchain4j-gpu-llama3 is not on the plugin classpath. Missing class: " + missing, t);
            }
            if (root instanceof NoClassDefFoundError) {
                String missing = root.getMessage() == null ? "" : root.getMessage();
                if (missing.contains("jdk/incubator/vector") || missing.contains("jdk.incubator.vector")) {
                    throw notify("GPULlama3 needs the Vector API. Add '--add-modules=jdk.incubator.vector' via Help → Edit Custom VM Options…, then restart the IDE.", t);
                }
                throw notify("GPULlama3 failed to load — likely missing TornadoVM. Details: " + rootMessage(t), t);
            }
            if (root instanceof OutOfMemoryError) {
                String msg = root.getMessage() == null ? "" : root.getMessage();
                if (msg.contains("direct buffer memory")) {
                    throw notify("GPULlama3 hit the JVM direct-memory limit while loading the GGUF tensors. Add '-XX:MaxDirectMemorySize=16g' (or higher, matching your model size) via Help → Edit Custom VM Options…, then restart.", t);
                }
                throw notify("GPULlama3 ran out of memory: " + rootMessage(t), t);
            }
            // GPU mode requires TornadoVM. Detect by stack-frame inspection because the
            // root is a vanilla NPE from Class.forName(null) inside TornadoVM's runtime
            // discovery — none of the type-based catches above will match it.
            if (stackMentions(t, "TornadoVMMasterPlan") || stackMentions(t, "TornadoRuntimeProvider")) {
                throw notify("GPU mode requires TornadoVM, which isn't available in this JVM. Uncheck 'GPULlama3 Use GPU' in Settings → Large Language Models for CPU-only mode.", t);
            }
            throw notify("GPULlama3 build failed: " + rootMessage(t), t);
        }
    }

    private static @NotNull Throwable root(@NotNull Throwable t) {
        Throwable cause = t;
        while (cause.getCause() != null && cause.getCause() != cause) {
            cause = cause.getCause();
        }
        return cause;
    }

    private static boolean stackMentions(@NotNull Throwable t, @NotNull String needle) {
        Throwable cur = t;
        while (cur != null) {
            for (StackTraceElement el : cur.getStackTrace()) {
                if (el.getClassName().contains(needle)) return true;
            }
            Throwable next = cur.getCause();
            if (next == null || next == cur) break;
            cur = next;
        }
        return false;
    }

    private static void invokeBuilder(@NotNull Object builder,
                                      @NotNull String method,
                                      @NotNull Class<?> paramType,
                                      @Nullable Object value) throws ReflectiveOperationException {
        builder.getClass().getMethod(method, paramType).invoke(builder, value);
    }

    private static @NotNull RuntimeException notify(@NotNull String message, @Nullable Throwable cause) {
        NotificationUtil.sendNotification(ProjectManager.getInstance().getDefaultProject(), message);
        return new RuntimeException(message, cause);
    }

    private static @NotNull String rootMessage(@NotNull Throwable t) {
        Throwable cause = root(t);
        return cause.getClass().getSimpleName() + (cause.getMessage() != null ? ": " + cause.getMessage() : "");
    }
}
