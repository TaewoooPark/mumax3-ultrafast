package engine

import (
	"net"
	"testing"
)

func TestListenForGUIReportsEphemeralPort(t *testing.T) {
	listener, requested := listenForGUI("127.0.0.1:0")
	defer listener.Close()

	advertised := advertisedGUIAddress(requested, listener.Addr())
	host, port, err := net.SplitHostPort(advertised)
	if err != nil {
		t.Fatalf("invalid advertised address %q: %v", advertised, err)
	}
	if host != "127.0.0.1" {
		t.Fatalf("advertised host = %q, want 127.0.0.1", host)
	}
	if port == "0" {
		t.Fatalf("advertised address contains an unresolved ephemeral port: %s", advertised)
	}
}

func TestAdvertisedGUIAddressPreservesFixedAddress(t *testing.T) {
	actual, err := net.ResolveTCPAddr("tcp", "[::]:35367")
	if err != nil {
		t.Fatal(err)
	}
	if got := advertisedGUIAddress(":35367", actual); got != ":35367" {
		t.Fatalf("advertised address = %q, want :35367", got)
	}
}
