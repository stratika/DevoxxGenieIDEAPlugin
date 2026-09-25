package com.devoxx.genie.chatmodel.local;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.ConnectException;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * What the user is told when a local provider's model probe fails.
 *
 * <p>The two causes need different answers: an unreachable endpoint is the user's to fix, a fault
 * on this side is not. Reporting the second as "LLM provider is not running" is what sent a real
 * debugging session to the server for an unregistered service — the server was answering
 * {@code /v1/models} in under a millisecond throughout.
 */
class LocalProviderFailureMessageTest {

    private static final String PROVIDER = "jitLLM";
    private static final String URL = "http://localhost:8090/v1/";

    @Test
    void anUnreachableEndpointNamesTheUrlAndTellsTheUserToStartIt() {
        String message = LocalChatModelFactory.providerUnavailableMessage(
                PROVIDER, URL, new ConnectException("Connection refused"));

        assertThat(message)
                .contains(PROVIDER)
                .contains(URL)
                .contains("Connection refused")
                .contains("Start it");
    }

    @Test
    void anUnreachableEndpointWithNoConfiguredUrlStillReadsSensibly() {
        assertThat(LocalChatModelFactory.providerUnavailableMessage(
                PROVIDER, null, new IOException("timeout")))
                .contains("its configured URL")
                .doesNotContain("null");
    }

    @Test
    void aBlankUrlIsTreatedAsUnconfigured() {
        assertThat(LocalChatModelFactory.providerUnavailableMessage(
                PROVIDER, "   ", new IOException("timeout")))
                .contains("its configured URL");
    }

    /**
     * The exact shape of the bug this was written for: an unregistered application service makes
     * its {@code @NotNull} accessor throw, which must not be reported as a stopped server.
     */
    @Test
    void aPluginSideFaultSaysSoAndDoesNotBlameTheServer() {
        String message = LocalChatModelFactory.providerUnavailableMessage(
                PROVIDER, URL,
                new IllegalStateException(
                        "@NotNull method JitLLMModelService.getInstance must not return null"));

        assertThat(message)
                .contains("IllegalStateException")
                .contains("JitLLMModelService")
                .contains("plugin-side error")
                .doesNotContain("Start it");
    }

    @Test
    void anExceptionWithoutAMessageFallsBackToItsType() {
        assertThat(LocalChatModelFactory.providerUnavailableMessage(
                PROVIDER, URL, new NullPointerException()))
                .contains("NullPointerException")
                .doesNotContain("null)");
    }

    @Test
    void theTwoCausesProduceDifferentAdvice() {
        String unreachable = LocalChatModelFactory.providerUnavailableMessage(
                PROVIDER, URL, new ConnectException("refused"));
        String internal = LocalChatModelFactory.providerUnavailableMessage(
                PROVIDER, URL, new IllegalStateException("boom"));

        assertThat(unreachable).isNotEqualTo(internal);
        assertThat(unreachable).doesNotContain("plugin-side");
        assertThat(internal).doesNotContain("correct the URL");
    }
}
