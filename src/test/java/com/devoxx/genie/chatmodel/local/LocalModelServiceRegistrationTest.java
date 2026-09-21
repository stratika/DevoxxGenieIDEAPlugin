package com.devoxx.genie.chatmodel.local;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Every {@code *ModelService} under {@code chatmodel/local} resolves itself with
 * {@code ApplicationManager.getApplication().getService(...)}, which returns {@code null} for a
 * class the plugin descriptor does not register. Because the accessor is annotated
 * {@code @NotNull}, the omission does not surface as a helpful message: the instrumented null
 * check throws {@code IllegalStateException: @NotNull method ... must not return null} from inside
 * the model fetch, which the provider panel reports to the user as the unrelated
 * "LLM provider is not running. Please start it and try again."
 *
 * <p>A unit test that constructs the service directly cannot catch this, so the registration
 * itself is asserted here — for every such service at once, so a newly added provider is covered
 * without anyone remembering to extend this test.
 */
class LocalModelServiceRegistrationTest {

    private static final Path LOCAL_PROVIDERS =
            Path.of("src/main/java/com/devoxx/genie/chatmodel/local");
    private static final Path PLUGIN_XML = Path.of("src/main/resources/META-INF/plugin.xml");

    private static List<String> localModelServiceClassNames() throws IOException {
        try (Stream<Path> paths = Files.walk(LOCAL_PROVIDERS)) {
            return paths.filter(Files::isRegularFile)
                    .map(Path::getFileName)
                    .map(Path::toString)
                    .filter(name -> name.endsWith("ModelService.java"))
                    .map(name -> name.substring(0, name.length() - ".java".length()))
                    .sorted()
                    .toList();
        }
    }

    @Test
    void everyLocalModelServiceIsRegisteredAsAnApplicationService() throws IOException {
        String pluginXml = Files.readString(PLUGIN_XML);
        List<String> services = localModelServiceClassNames();

        assertThat(services)
                .as("the scan must actually find the local model services")
                .isNotEmpty();

        for (String simpleName : services) {
            assertThat(pluginXml)
                    .as("plugin.xml must register %s as an <applicationService>, "
                            + "otherwise getService() returns null at runtime", simpleName)
                    .contains(".chatmodel.local.")
                    .containsPattern("serviceImplementation=\"com\\.devoxx\\.genie\\.chatmodel\\.local\\."
                            + "[a-z0-9]+\\." + simpleName + "\"");
        }
    }

    /** The service this test was written for, named explicitly so a regression is unambiguous. */
    @Test
    void gpuLlama3ModelServiceIsRegistered() throws IOException {
        assertThat(Files.readString(PLUGIN_XML))
                .contains("com.devoxx.genie.chatmodel.local.gpullama3.GPULlama3ModelService");
    }
}
