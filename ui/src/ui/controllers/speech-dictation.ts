type DictationCallbacks = {
  onListeningChange: (listening: boolean) => void;
  onPartial: (text: string) => void;
  onFinal: (text: string) => void;
  onError: (message: string) => void;
};

type SpeechRecognitionResultLike = {
  isFinal: boolean;
  0: { transcript: string };
};

type SpeechRecognitionEventLike = Event & {
  resultIndex: number;
  results: ArrayLike<SpeechRecognitionResultLike>;
};

type SpeechRecognitionLike = EventTarget & {
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  lang: string;
  onstart: (() => void) | null;
  onend: (() => void) | null;
  onerror: ((event: Event & { error?: string }) => void) | null;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  start: () => void;
  stop: () => void;
};

type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

export type SpeechDictationController = {
  isSupported: () => boolean;
  isListening: () => boolean;
  start: () => boolean;
  stop: () => void;
  toggle: () => boolean;
  dispose: () => void;
};

function getRecognitionCtor(): SpeechRecognitionCtor | null {
  if (typeof window === "undefined") {
    return null;
  }
  const maybeCtor = (window as unknown as Record<string, unknown>).SpeechRecognition;
  if (typeof maybeCtor === "function") {
    return maybeCtor as SpeechRecognitionCtor;
  }
  const maybeWebkitCtor = (window as unknown as Record<string, unknown>).webkitSpeechRecognition;
  if (typeof maybeWebkitCtor === "function") {
    return maybeWebkitCtor as SpeechRecognitionCtor;
  }
  return null;
}

function mapSpeechError(error?: string): string {
  switch (error) {
    case "not-allowed":
    case "service-not-allowed":
      return "Microphone permission is blocked. Allow mic access and try again.";
    case "audio-capture":
      return "No microphone detected.";
    case "network":
      return "Speech recognition network error.";
    case "no-speech":
      return "No speech detected.";
    default:
      return "Speech recognition failed.";
  }
}

export function createSpeechDictationController(
  callbacks: DictationCallbacks,
): SpeechDictationController {
  let recognition: SpeechRecognitionLike | null = null;
  let listening = false;
  let onErrorHandler: ((event: Event) => void) | null = null;

  const ensureRecognition = (): SpeechRecognitionLike | null => {
    if (recognition) {
      return recognition;
    }
    const Ctor = getRecognitionCtor();
    if (!Ctor) {
      return null;
    }
    const next = new Ctor();
    next.continuous = true;
    next.interimResults = true;
    next.maxAlternatives = 1;
    next.lang = "en-US";
    next.onstart = () => {
      listening = true;
      callbacks.onListeningChange(true);
    };
    next.onend = () => {
      listening = false;
      callbacks.onListeningChange(false);
      callbacks.onPartial("");
    };
    onErrorHandler = (event: Event) => {
      const withError = event as Event & { error?: string };
      callbacks.onError(mapSpeechError(withError.error));
    };
    next.addEventListener("error", onErrorHandler);
    next.onresult = (event) => {
      let partial = "";
      let finalText = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        const transcript = result?.[0]?.transcript ?? "";
        if (!transcript) {
          continue;
        }
        if (result.isFinal) {
          finalText += transcript;
        } else {
          partial += transcript;
        }
      }
      callbacks.onPartial(partial.trim());
      if (finalText.trim()) {
        callbacks.onFinal(finalText.trim());
      }
    };
    recognition = next;
    return next;
  };

  const start = (): boolean => {
    if (listening) {
      return true;
    }
    const instance = ensureRecognition();
    if (!instance) {
      callbacks.onError("Speech recognition is not supported in this browser.");
      return false;
    }
    try {
      callbacks.onError("");
      instance.start();
      return true;
    } catch {
      callbacks.onError("Unable to start microphone dictation.");
      return false;
    }
  };

  const stop = () => {
    if (!recognition || !listening) {
      return;
    }
    recognition.stop();
  };

  return {
    isSupported: () => Boolean(getRecognitionCtor()),
    isListening: () => listening,
    start,
    stop,
    toggle: () => {
      if (listening) {
        stop();
        return false;
      }
      return start();
    },
    dispose: () => {
      if (recognition) {
        if (onErrorHandler) {
          recognition.removeEventListener("error", onErrorHandler);
          onErrorHandler = null;
        }
        recognition.onstart = null;
        recognition.onend = null;
        recognition.onresult = null;
      }
      recognition = null;
      listening = false;
      callbacks.onListeningChange(false);
      callbacks.onPartial("");
    },
  };
}
