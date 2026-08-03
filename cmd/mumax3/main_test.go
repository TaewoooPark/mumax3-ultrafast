package main

import "testing"

func TestGUIURL(t *testing.T) {
	tests := map[string]string{
		":35367":          "http://127.0.0.1:35367",
		"127.0.0.1:35367": "http://127.0.0.1:35367",
		"[::1]:35367":     "http://[::1]:35367",
		"localhost:35367": "http://localhost:35367",
	}
	for address, want := range tests {
		if got := guiURL(address); got != want {
			t.Errorf("guiURL(%q) = %q, want %q", address, got, want)
		}
	}
}
