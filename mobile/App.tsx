import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  AppState,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from 'react-native';
import { Image } from 'expo-image';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import {
  QueryClient,
  QueryClientProvider,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import {
  SpaceGrotesk_500Medium,
  SpaceGrotesk_700Bold,
  useFonts as useSpaceGrotesk,
} from '@expo-google-fonts/space-grotesk';
import {
  Manrope_400Regular,
  Manrope_600SemiBold,
  Manrope_700Bold,
  useFonts as useManrope,
} from '@expo-google-fonts/manrope';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import {
  createSession,
  getManifest,
  getSessionState,
  getStats,
  playRound,
} from './src/features/game/api';
import { useGameStore } from './src/store/useGameStore';
import { theme } from './src/theme/tokens';
import { toBasicAuthHeader } from './src/lib/http';
import {
  clearApiBaseUrlOverride,
  DEFAULT_API_BASE_URL,
  getApiBaseUrl,
  MANUAL_API_ORIGIN_ENABLED,
  setApiBaseUrlOverride,
} from './src/config/env';
import { NeonButton } from './src/ui/NeonButton';
import { NeonCard } from './src/ui/NeonCard';
import { transcriptToMove, type GameMove } from './src/features/audio/transcript';
import {
  isSpeechRecognitionAvailable,
  requestSpeechPermissions,
  startSpeechRecognition,
  stopSpeechRecognition,
  useSafeSpeechRecognitionEvent,
} from './src/features/audio/speech';
import {
  inferMoveFromImage,
  type VisionPrediction,
} from './src/features/vision/inference';
import type { InputMode, PlayRoundResponse } from './src/features/game/schemas';
import { classifyAppError, type AppErrorScreen } from './src/features/game/errors';
import { deriveInputModeSupport } from './src/features/game/inputModes';
import { latestRoundFromSession, sessionStateToRounds } from './src/features/game/session';
import {
  clearAuthSession,
  clearSavedApiBaseUrl,
  clearSavedSession,
  loadApiBaseUrl,
  loadAuthSession,
  loadInputMode,
  loadSession,
  saveApiBaseUrl,
  saveAuthSession,
  saveInputMode,
  saveSession,
} from './src/infra/preferences';

const queryClient = new QueryClient();

type LoginForm = {
  username: string;
  password: string;
};

type PrimaryTab = 'arena' | 'history' | 'stats' | 'settings';

type ErrorModel = {
  heading: string;
  subtitle: string;
  code: string;
  icon: keyof typeof MaterialCommunityIcons.glyphMap;
  primaryCta: string;
  secondaryCta?: string;
};

const moveOptions: Array<{
  move: GameMove;
  label: string;
  caption: string;
  icon: keyof typeof MaterialCommunityIcons.glyphMap;
}> = [
  { move: 'rock', label: 'Rock', caption: 'Hold the line', icon: 'octagon-outline' },
  { move: 'paper', label: 'Paper', caption: 'Cover the board', icon: 'file-document-outline' },
  { move: 'scissors', label: 'Scissors', caption: 'Sharp counter', icon: 'content-cut' },
  { move: 'none', label: 'Pass', caption: 'Let server randomize', icon: 'shuffle-variant' },
];

function AppScreen() {
  const queryClientHook = useQueryClient();
  const [activeTab, setActiveTab] = useState<PrimaryTab>('arena');
  const [overlayScreen, setOverlayScreen] = useState<AppErrorScreen | null>(null);
  const [bootstrapping, setBootstrapping] = useState(true);
  const [form, setForm] = useState<LoginForm>({ username: '', password: '' });
  const [apiBaseUrl, setApiBaseUrl] = useState(() => getApiBaseUrl());
  const [apiBaseUrlDraft, setApiBaseUrlDraft] = useState(() => getApiBaseUrl());
  const [apiBaseUrlError, setApiBaseUrlError] = useState<string | null>(null);
  const [loginError, setLoginError] = useState<string | null>(null);
  const [visionError, setVisionError] = useState<string | null>(null);
  const [audioError, setAudioError] = useState<string | null>(null);
  const [visionPreviewUri, setVisionPreviewUri] = useState<string | null>(null);
  const [visionPrediction, setVisionPrediction] = useState<VisionPrediction | null>(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [recognizing, setRecognizing] = useState(false);
  const [visionWorking, setVisionWorking] = useState(false);
  const [hapticsEnabled, setHapticsEnabled] = useState(true);
  const [roundHistory, setRoundHistory] = useState<PlayRoundResponse[]>([]);
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView | null>(null);

  const {
    authHeader,
    username,
    session,
    selectedMode,
    busy,
    lastRound,
    setAuth,
    clearAuth,
    setSession,
    clearSession,
    setSelectedMode,
    setBusy,
    setLastRound,
  } = useGameStore();

  const statsQuery = useQuery({
    queryKey: ['stats', authHeader],
    queryFn: () => getStats(authHeader as string),
    enabled: Boolean(authHeader),
  });

  const manifestQuery = useQuery({
    queryKey: ['manifest', authHeader],
    queryFn: () => getManifest(authHeader as string),
    enabled: Boolean(authHeader),
  });

  const manifest = manifestQuery.data;
  const modeSupport = useMemo(
    () => deriveInputModeSupport(manifest, { speechAvailable: isSpeechRecognitionAvailable }),
    [manifest],
  );
  const { availableModes, visionModelAvailable, audioSpeechAvailable, visionEnabled, audioEnabled } = modeSupport;
  const isModeReady = (mode: InputMode) => {
    if (mode === 'vision') {
      return visionEnabled;
    }
    if (mode === 'audio') {
      return audioEnabled;
    }
    return availableModes.includes('buttons');
  };

  const transcriptMove = useMemo(() => transcriptToMove(transcript), [transcript]);

  const scoreSnapshot = useMemo(() => {
    const player = lastRound?.player_score ?? 0;
    const house = lastRound?.server_score ?? 0;
    return { player, house };
  }, [lastRound]);

  const statsSnapshot = useMemo(() => {
    const wins = statsQuery.data?.matches_won ?? 0;
    const losses = statsQuery.data?.matches_lost ?? 0;
    const draws = statsQuery.data?.matches_draw ?? 0;
    const rounds = statsQuery.data?.rounds_played ?? 0;
    const completed = statsQuery.data?.matches_completed ?? 0;
    const winRate = completed > 0 ? (wins / completed) * 100 : 0;

    return {
      wins,
      losses,
      draws,
      rounds,
      completed,
      sessionsStarted: statsQuery.data?.sessions_started ?? 0,
      winRate,
    };
  }, [statsQuery.data]);

  const matchLocked = busy || visionWorking || !session || Boolean(lastRound?.match_complete);
  const showMatchSummary = Boolean(lastRound?.match_complete);
  const matchSummaryTitle =
    lastRound?.winner === 'player'
      ? 'Player wins'
      : lastRound?.winner === 'server'
        ? 'House wins'
        : lastRound?.winner === 'draw'
          ? 'Draw'
          : 'Match complete';
  const matchSummarySubtitle =
    lastRound?.winner === 'player'
      ? 'You closed the match ahead of the house model.'
      : lastRound?.winner === 'server'
        ? 'The house model finished ahead this round.'
        : lastRound?.winner === 'draw'
          ? 'Both sides finished on the same scoreline.'
          : 'The current match is complete.';

  const errorModel = useMemo<ErrorModel | null>(() => {
    if (!overlayScreen) {
      return null;
    }

    if (overlayScreen === 'auth_error') {
      return {
        heading: 'Authorization failed',
        subtitle: 'Check the username and password, then start a fresh session.',
        code: '401 Unauthorized',
        icon: 'shield-alert-outline',
        primaryCta: 'Log in again',
      };
    }

    if (overlayScreen === 'network_failure') {
      return {
        heading: 'Connection unavailable',
        subtitle: 'The app could not reach the backend service. Try reconnecting.',
        code: 'Network request failed',
        icon: 'wifi-off',
        primaryCta: 'Reconnect',
      };
    }

    if (overlayScreen === 'session_expired') {
      return {
        heading: 'Session expired',
        subtitle: 'This match session is no longer active. Start a new match to continue.',
        code: 'Session inactive',
        icon: 'timer-sand',
        primaryCta: 'Start new match',
        secondaryCta: 'Return to arena',
      };
    }

    if (overlayScreen === 'match_complete') {
      return {
        heading: 'Match already complete',
        subtitle: 'This session is locked. Start a new match to continue playing.',
        code: 'Match complete',
        icon: 'flag-checkered',
        primaryCta: 'Start new match',
        secondaryCta: 'Return to arena',
      };
    }

    if (overlayScreen === 'max_sessions_reached') {
      return {
        heading: 'Session limit reached',
        subtitle: 'Too many active sessions are open for this account.',
        code: 'Active sessions exceeded',
        icon: 'account-multiple-remove-outline',
        primaryCta: 'Back to settings',
      };
    }

    return {
      heading: 'Model unavailable',
      subtitle: 'The selected input model is not available in the current manifest.',
      code: 'Model not deployed',
      icon: 'memory',
      primaryCta: 'Return to arena',
    };
  }, [overlayScreen]);

  useEffect(() => {
    let active = true;

    void (async () => {
      try {
        const [savedApiBaseUrl, savedMode, savedAuth, savedSession] = await Promise.all([
          MANUAL_API_ORIGIN_ENABLED ? loadApiBaseUrl() : Promise.resolve(null),
          loadInputMode(),
          loadAuthSession(),
          loadSession(),
        ]);

        if (!active) {
          return;
        }

        let restoredApiBaseUrl = getApiBaseUrl();
        if (!MANUAL_API_ORIGIN_ENABLED) {
          clearApiBaseUrlOverride();
          await clearSavedApiBaseUrl();
          restoredApiBaseUrl = getApiBaseUrl();
        } else if (savedApiBaseUrl) {
          try {
            restoredApiBaseUrl = setApiBaseUrlOverride(savedApiBaseUrl);
          } catch {
            await clearSavedApiBaseUrl();
            restoredApiBaseUrl = getApiBaseUrl();
          }
        }
        setApiBaseUrl(restoredApiBaseUrl);
        setApiBaseUrlDraft(restoredApiBaseUrl);
        setApiBaseUrlError(null);

        if (savedMode) {
          setSelectedMode(savedMode);
        }

        if (!savedAuth) {
          return;
        }

        setAuth(savedAuth.authHeader, savedAuth.username);

        if (savedSession) {
          try {
            const state = await getSessionState(savedAuth.authHeader, savedSession.sessionId);
            if (!active) {
              return;
            }
            setSession(state.session_id, state.max_rounds);
            const rounds = sessionStateToRounds(state);
            setRoundHistory(rounds);
            setLastRound(latestRoundFromSession(state));
            return;
          } catch (error) {
            if (!active) {
              return;
            }
            const next = classifyAppError(error);
            if (next === 'auth_error') {
              clearAuth();
              await clearAuthSession();
              return;
            }
            if (next === 'session_expired') {
              clearSession();
              await clearSavedSession();
            }
          }
        }

        const created = await createSession(savedAuth.authHeader);
        if (!active) {
          return;
        }
        setSession(created.session_id, created.max_rounds);
        await saveSession(created.session_id, created.max_rounds);
      } catch {
        if (active) {
          clearAuth();
          await clearAuthSession();
        }
      } finally {
        if (active) {
          setBootstrapping(false);
        }
      }
    })();

    return () => {
      active = false;
    };
  }, [clearAuth, setAuth, setLastRound, setSelectedMode, setSession]);

  useEffect(() => {
    if (!availableModes.includes(selectedMode)) {
      setSelectedMode('buttons');
    }
  }, [availableModes, selectedMode, setSelectedMode]);

  useEffect(() => {
    void saveInputMode(selectedMode);
  }, [selectedMode]);

  useEffect(() => {
    if (selectedMode === 'vision' && !visionEnabled) {
      setOverlayScreen('model_unavailable');
      setSelectedMode('buttons');
    }
    if (selectedMode === 'audio' && !audioEnabled) {
      setOverlayScreen('model_unavailable');
      setSelectedMode('buttons');
    }
  }, [audioEnabled, selectedMode, setSelectedMode, visionEnabled]);

  useEffect(() => {
    if (selectedMode !== 'vision') {
      setCameraOpen(false);
    }
  }, [selectedMode]);

  useEffect(() => {
    if (!statsQuery.error) {
      return;
    }
    const next = classifyAppError(statsQuery.error);
    if (next) {
      setOverlayScreen(next);
    }
  }, [statsQuery.error]);

  useEffect(() => {
    if (!manifestQuery.error) {
      return;
    }
    const next = classifyAppError(manifestQuery.error);
    if (next) {
      setOverlayScreen(next);
    }
  }, [manifestQuery.error]);

  useEffect(() => {
    const subscription = AppState.addEventListener('change', (state) => {
      if (state !== 'active' || !authHeader || !session?.sessionId) {
        return;
      }

      void (async () => {
        try {
          const stateSnapshot = await getSessionState(authHeader, session.sessionId);
          setSession(stateSnapshot.session_id, stateSnapshot.max_rounds);
          setRoundHistory(sessionStateToRounds(stateSnapshot));
          setLastRound(latestRoundFromSession(stateSnapshot));
        } catch (error) {
          const next = classifyAppError(error);
          if (!next) {
            return;
          }
          setOverlayScreen(next);
          if (next === 'session_expired') {
            clearSession();
          }
          if (next === 'auth_error') {
            clearAuth();
            void clearAuthSession();
          }
        }
      })();
    });

    return () => {
      subscription.remove();
    };
  }, [authHeader, clearAuth, clearSession, session?.sessionId, setLastRound, setSession]);

  useSafeSpeechRecognitionEvent('start', () => {
    setRecognizing(true);
    setAudioError(null);
  });

  useSafeSpeechRecognitionEvent('end', () => {
    setRecognizing(false);
  });

  useSafeSpeechRecognitionEvent('result', (event: any) => {
    const last = event?.results?.[0]?.transcript ?? event?.results?.[0]?.[0]?.transcript;
    if (typeof last === 'string') {
      setTranscript(last);
      const move = transcriptToMove(last);
      if (move && !matchLocked) {
        playMutation.mutate(move);
      } else if (!move) {
        setAudioError(`Say rock, paper, scissors, or none. Heard: ${last || '(empty)'}`);
      }
    }
  });

  useSafeSpeechRecognitionEvent('error', (event: any) => {
    setRecognizing(false);
    setAudioError(event?.message ?? 'Speech recognition failed');
  });

  const loginMutation = useMutation({
    mutationFn: async ({ username: user, password }: LoginForm) => {
      const header = toBasicAuthHeader(user.trim(), password);
      const created = await createSession(header);
      return { header, user: user.trim(), created };
    },
    onMutate: async () => {
      setBusy(true);
      setLoginError(null);
      setOverlayScreen(null);
      clearSession();
      await clearSavedSession();
    },
    onSuccess: async ({ header, user, created }) => {
      setAuth(header, user);
      setSession(created.session_id, created.max_rounds);
      await saveAuthSession(header, user);
      await saveSession(created.session_id, created.max_rounds);
      setLastRound(null);
      setRoundHistory([]);
      await queryClientHook.invalidateQueries({ queryKey: ['stats'] });
      await queryClientHook.invalidateQueries({ queryKey: ['manifest'] });
      setOverlayScreen(null);
      setActiveTab('arena');
    },
    onError: (error) => {
      const message = error instanceof Error ? error.message : 'Unable to login';
      setLoginError(message);
      clearAuth();
      clearSession();
      void clearAuthSession();
      setRoundHistory([]);
      const next = classifyAppError(error);
      if (next) {
        setOverlayScreen(next);
      }
    },
    onSettled: () => setBusy(false),
  });

  const newMatchMutation = useMutation({
    mutationFn: async () => {
      if (!authHeader) {
        throw new Error('Not authenticated');
      }
      return createSession(authHeader);
    },
    onMutate: () => {
      setBusy(true);
      setOverlayScreen(null);
    },
    onSuccess: async (created) => {
      setSession(created.session_id, created.max_rounds);
      await saveSession(created.session_id, created.max_rounds);
      setLastRound(null);
      setRoundHistory([]);
      setVisionPreviewUri(null);
      setVisionPrediction(null);
      setTranscript('');
      await queryClientHook.invalidateQueries({ queryKey: ['stats'] });
      await queryClientHook.invalidateQueries({ queryKey: ['manifest'] });
      setActiveTab('arena');
    },
    onError: (error) => {
      setLoginError(error instanceof Error ? error.message : 'Unable to start new match');
      const next = classifyAppError(error);
      if (next) {
        setOverlayScreen(next);
      }
    },
    onSettled: () => setBusy(false),
  });

  const playMutation = useMutation({
    mutationFn: async (move: GameMove) => {
      if (!authHeader || !session?.sessionId) {
        throw new Error('No active session');
      }
      return playRound(authHeader, session.sessionId, move);
    },
    onMutate: () => {
      setBusy(true);
      setOverlayScreen(null);
    },
    onSuccess: async (roundResult) => {
      setLastRound(roundResult);
      setRoundHistory((current) => [...current, roundResult]);
      if (hapticsEnabled) {
        await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
      }
      if (roundResult.match_complete) {
        await queryClientHook.invalidateQueries({ queryKey: ['stats'] });
      }
    },
    onError: (error) => {
      setLoginError(error instanceof Error ? error.message : 'Unable to play round');
      const next = classifyAppError(error);
      if (next) {
        setOverlayScreen(next);
      }
      if (next === 'session_expired') {
        clearSession();
      }
      if (next === 'auth_error') {
        clearAuth();
        void clearAuthSession();
      }
      if (hapticsEnabled) {
        void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
      }
    },
    onSettled: () => setBusy(false),
  });

  const handleModeSelection = (mode: InputMode) => {
    if (mode === 'vision' && !visionEnabled) {
      setOverlayScreen('model_unavailable');
      return;
    }
    if (mode === 'audio' && !audioEnabled) {
      setOverlayScreen('model_unavailable');
      return;
    }
    setSelectedMode(mode);
    setActiveTab('arena');
  };

  const clearNetworkBoundState = async () => {
    clearAuth();
    clearSession();
    setLastRound(null);
    setRoundHistory([]);
    setVisionPrediction(null);
    setVisionPreviewUri(null);
    setTranscript('');
    setOverlayScreen(null);
    queryClientHook.clear();
    await Promise.all([clearAuthSession(), clearSavedSession()]);
  };

  const commitApiBaseUrlDraft = async (): Promise<string | null> => {
    if (!MANUAL_API_ORIGIN_ENABLED) {
      const fixedBaseUrl = getApiBaseUrl();
      setApiBaseUrl(fixedBaseUrl);
      setApiBaseUrlDraft(fixedBaseUrl);
      setApiBaseUrlError(null);
      return fixedBaseUrl;
    }

    try {
      const normalized = setApiBaseUrlOverride(apiBaseUrlDraft);
      const changed = normalized !== apiBaseUrl;
      setApiBaseUrl(normalized);
      setApiBaseUrlDraft(normalized);
      setApiBaseUrlError(null);
      await saveApiBaseUrl(normalized);
      if (changed) {
        await clearNetworkBoundState();
      }
      return normalized;
    } catch (error) {
      setApiBaseUrlError(error instanceof Error ? error.message : 'Invalid API origin');
      return null;
    }
  };

  const handleResetApiBaseUrl = async () => {
    clearApiBaseUrlOverride();
    const fallback = getApiBaseUrl();
    setApiBaseUrl(fallback);
    setApiBaseUrlDraft(fallback);
    setApiBaseUrlError(null);
    await clearSavedApiBaseUrl();
    await clearNetworkBoundState();
  };

  const handleLogin = () => {
    if (!form.username.trim() || !form.password) {
      setLoginError('Username and password are required');
      return;
    }

    void (async () => {
      const normalized = await commitApiBaseUrlDraft();
      if (!normalized) {
        return;
      }
      loginMutation.mutate({ username: form.username.trim(), password: form.password });
    })();
  };

  const handleLogout = () => {
    clearAuth();
    clearSession();
    void clearAuthSession();
    queryClientHook.removeQueries({ queryKey: ['stats'] });
    queryClientHook.removeQueries({ queryKey: ['manifest'] });
    setForm({ username: '', password: '' });
    setLoginError(null);
    setVisionPrediction(null);
    setVisionPreviewUri(null);
    setTranscript('');
    setRoundHistory([]);
    setOverlayScreen(null);
    setActiveTab('arena');
  };

  const playSelectedMove = (move: GameMove) => {
    if (matchLocked) {
      return;
    }
    playMutation.mutate(move);
  };

  const runVisionInference = async (uri: string, fileName?: string) => {
    try {
      if (!manifest || !authHeader) {
        setVisionError('ML manifest is not ready yet');
        return;
      }

      setVisionWorking(true);
      setVisionError(null);
      const prediction = await inferMoveFromImage(uri, manifest, authHeader, fileName);
      setVisionPreviewUri(uri);
      setVisionPrediction(prediction);
      if (hapticsEnabled) {
        await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
      }
      if (!lastRound?.match_complete && session?.sessionId) {
        playMutation.mutate(prediction.move);
      }
    } catch (error) {
      setVisionError(error instanceof Error ? error.message : 'Vision inference failed');
    } finally {
      setVisionWorking(false);
    }
  };

  const openCameraCapture = async () => {
    if (!cameraPermission?.granted) {
      const result = await requestCameraPermission();
      if (!result.granted) {
        setVisionError('Camera permission is required for live capture mode');
        return;
      }
    }
    setVisionError(null);
    setCameraOpen(true);
  };

  const captureSingleFrame = async () => {
    if (!cameraRef.current) {
      return;
    }

    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.8 });
      if (!photo?.uri) {
        throw new Error('Capture failed');
      }
      setCameraOpen(false);
      await runVisionInference(photo.uri);
    } catch (error) {
      setVisionError(error instanceof Error ? error.message : 'Unable to capture frame');
    }
  };

  const pickImageAndInfer = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setVisionError('Media library permission is required to pick an image');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 1,
    });

    if (result.canceled) {
      return;
    }

    const asset = result.assets[0];
    if (!asset?.uri) {
      return;
    }

    await runVisionInference(asset.uri, asset.fileName ?? undefined);
  };

  const startListening = async () => {
    try {
      setAudioError(null);
      if (!isSpeechRecognitionAvailable) {
        setAudioError('Speech recognition is not available in this runtime. Use a development build or web.');
        return;
      }

      const granted = await requestSpeechPermissions();
      if (!granted) {
        setAudioError('Microphone permissions were denied');
        return;
      }

      const started = startSpeechRecognition({
        lang: manifest?.audio?.browser_speech?.locale ?? 'en-US',
        interimResults: false,
        continuous: false,
        maxAlternatives: 1,
      });

      if (!started) {
        setAudioError('Unable to start speech recognition in this runtime');
      }
    } catch (error) {
      setAudioError(error instanceof Error ? error.message : 'Unable to start speech recognition');
    }
  };

  const stopListening = () => {
    stopSpeechRecognition();
  };

  const handleOverlayPrimary = () => {
    if (!overlayScreen) {
      return;
    }

    if (overlayScreen === 'auth_error') {
      handleLogout();
      return;
    }

    if (overlayScreen === 'network_failure') {
      setOverlayScreen(null);
      void queryClientHook.invalidateQueries({ queryKey: ['stats'] });
      void queryClientHook.invalidateQueries({ queryKey: ['manifest'] });
      return;
    }

    if (overlayScreen === 'session_expired') {
      setOverlayScreen(null);
      if (authHeader) {
        newMatchMutation.mutate();
      }
      return;
    }

    if (overlayScreen === 'match_complete') {
      setOverlayScreen(null);
      if (authHeader) {
        newMatchMutation.mutate();
      }
      return;
    }

    if (overlayScreen === 'max_sessions_reached') {
      setOverlayScreen(null);
      return;
    }

    setOverlayScreen(null);
    setSelectedMode('buttons');
  };

  const handleOverlaySecondary = () => {
    setOverlayScreen(null);
    setActiveTab('arena');
  };

  const renderSystemState = () => {
    if (!errorModel) {
      return null;
    }

    return (
      <SafeAreaView style={styles.safe}>
        <StatusBar style="light" />
        <LinearGradient
          colors={[theme.colors.surface, theme.colors.surfaceContainer, theme.colors.surface]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.errorRoot}
        >
          <View style={styles.errorCard}>
            <View style={styles.errorBadgeShell}>
              <MaterialCommunityIcons name={errorModel.icon} size={46} color={theme.colors.error} />
            </View>
            <Text style={styles.errorHeading}>{errorModel.heading}</Text>
            <Text style={styles.errorSubheading}>{errorModel.subtitle}</Text>
            <Text style={styles.errorCode}>Reference: {errorModel.code}</Text>
            <NeonButton label={errorModel.primaryCta} onPress={handleOverlayPrimary} />
            {errorModel.secondaryCta ? (
              <NeonButton label={errorModel.secondaryCta} variant="secondary" onPress={handleOverlaySecondary} />
            ) : null}
          </View>
        </LinearGradient>
      </SafeAreaView>
    );
  };

  if (bootstrapping) {
    return (
      <SafeAreaView style={styles.safe}>
        <StatusBar style="light" />
        <View style={styles.loadingScreen}>
          <ActivityIndicator color={theme.colors.primary} size="large" />
          <Text style={styles.loadingText}>Restoring secure session...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (!authHeader) {
    if (overlayScreen && overlayScreen !== 'model_unavailable') {
      return renderSystemState();
    }

    return (
      <SafeAreaView style={styles.safe}>
        <StatusBar style="light" />
        <LinearGradient
          colors={[theme.colors.surface, theme.colors.surfaceContainerLow, theme.colors.surface]}
          start={{ x: 0, y: 0 }}
          end={{ x: 0.9, y: 1 }}
          style={styles.loginGradient}
        >
          <View style={styles.loginShell}>
            <View style={styles.loginMarkRow}>
              <View style={styles.appMark}>
                <MaterialCommunityIcons name="gesture-tap-button" size={26} color={theme.colors.onPrimary} />
              </View>
              <View style={styles.loginStatusPill}>
                <View style={styles.liveDot} />
                <Text style={styles.loginStatusText}>Game server link</Text>
              </View>
            </View>
            <Text style={styles.title}>RPS Arena</Text>
            <Text style={styles.subtitle}>
              Sign in to start a match, sync model settings, and play with button, camera, or voice input.
            </Text>

            <NeonCard style={styles.loginCard} tone="accent">
              {MANUAL_API_ORIGIN_ENABLED ? (
                <>
                  <Text style={styles.fieldLabel}>API Origin</Text>
                  <TextInput
                    autoCapitalize="none"
                    autoCorrect={false}
                    keyboardType="url"
                    value={apiBaseUrlDraft}
                    onChangeText={(value) => {
                      setApiBaseUrlDraft(value);
                      setApiBaseUrlError(null);
                    }}
                    style={styles.input}
                    placeholder="https://your-backend.example.com"
                    placeholderTextColor={theme.colors.onSurfaceMuted}
                  />
                  {apiBaseUrlError ? <Text style={styles.errorInlineText}>{apiBaseUrlError}</Text> : null}
                </>
              ) : null}

              <Text style={styles.fieldLabel}>Access ID</Text>
              <TextInput
                autoCapitalize="none"
                autoCorrect={false}
                value={form.username}
                onChangeText={(value) => setForm((current) => ({ ...current, username: value }))}
                style={styles.input}
                placeholder="Username"
                placeholderTextColor={theme.colors.onSurfaceMuted}
              />

              <Text style={styles.fieldLabel}>Security Key</Text>
              <TextInput
                secureTextEntry
                value={form.password}
                onChangeText={(value) => setForm((current) => ({ ...current, password: value }))}
                style={styles.input}
                placeholder="Password"
                placeholderTextColor={theme.colors.onSurfaceMuted}
              />

              {loginError ? <Text style={styles.errorInlineText}>{loginError}</Text> : null}

              <NeonButton
                label={busy ? 'Signing in...' : 'Start session'}
                icon="login-variant"
                onPress={handleLogin}
                disabled={busy}
              />
            </NeonCard>
            <Text style={styles.loginFootnote}>
              {MANUAL_API_ORIGIN_ENABLED
                ? 'Credentials are stored with SecureStore when available.'
                : 'The production backend is already configured for this build.'}
            </Text>
          </View>
        </LinearGradient>
      </SafeAreaView>
    );
  }

  if (overlayScreen) {
    return renderSystemState();
  }

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="light" />
      <LinearGradient
        colors={[theme.colors.surface, theme.colors.surfaceContainerLow, theme.colors.surface]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.appGradient}
      >
        <BlurView intensity={42} tint="dark" style={styles.topBar}>
          <View style={styles.avatarChip}>
            <Text style={styles.avatarInitial}>{(username?.[0] ?? 'P').toUpperCase()}</Text>
          </View>
          <View style={styles.topBarTitleBlock}>
            <Text style={styles.topBarTitle}>RPS Arena</Text>
            <Text style={styles.topBarSubTitle}>Round {Math.min(roundHistory.length + 1, session?.maxRounds ?? 1)} of {session?.maxRounds ?? 0}</Text>
          </View>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Open settings"
            onPress={() => setActiveTab('settings')}
            style={styles.settingsButton}
          >
            <MaterialCommunityIcons name="tune-variant" size={24} color={theme.colors.primary} />
          </Pressable>
        </BlurView>

        {activeTab === 'arena' ? (
          <ScrollView contentContainerStyle={styles.tabContent}>
            <NeonCard style={styles.scoreCard} tone="accent">
              <View style={styles.scoreColumns}>
                <View>
                  <Text style={styles.scoreLabel}>Player</Text>
                  <Text style={styles.scoreValue}>{String(scoreSnapshot.player).padStart(2, '0')}</Text>
                </View>
                <View style={styles.scoreCenterCol}>
                  <Text style={styles.roundLabel}>Round {roundHistory.length + 1}/{session?.maxRounds ?? 0}</Text>
                  <Text style={styles.versusText}>VS</Text>
                </View>
                <View style={styles.rightScore}>
                  <Text style={styles.scoreLabel}>House AI</Text>
                  <Text style={[styles.scoreValue, styles.houseScore]}>{String(scoreSnapshot.house).padStart(2, '0')}</Text>
                </View>
              </View>
            </NeonCard>

            {!showMatchSummary ? (
              <>
                <NeonCard style={styles.hubCard} tone="flat">
                  <View style={styles.hubTitleRow}>
                    <View>
                      <Text style={styles.cardEyebrow}>Current mode</Text>
                      <Text style={styles.cardTitle}>Match control</Text>
                    </View>
                    <View style={styles.liveChip}>
                      <View style={styles.liveDot} />
                      <Text style={styles.liveChipText}>Live</Text>
                    </View>
                  </View>
                  <Text style={styles.bodyText}>Input modes, model slot, and session rules are synced from the server manifest.</Text>
                  <View style={styles.modeRow}>
                    {(['buttons', 'vision', 'audio'] as InputMode[]).map((mode) => {
                      const isEnabled = isModeReady(mode);
                      const active = selectedMode === mode;
                      const icon =
                        mode === 'buttons'
                          ? 'gesture-tap-button'
                          : mode === 'vision'
                            ? 'camera-iris'
                            : 'microphone-outline';
                      return (
                        <Pressable
                          key={mode}
                          accessibilityRole="button"
                          accessibilityLabel={`Use ${mode} input mode${isEnabled ? '' : ', unavailable'}`}
                          accessibilityState={{ selected: active }}
                          onPress={() => handleModeSelection(mode)}
                          style={({ pressed }) => [
                            styles.modeChip,
                            active && styles.modeChipActive,
                            !isEnabled && styles.modeChipDisabled,
                            pressed && styles.pressed,
                          ]}
                        >
                          <MaterialCommunityIcons
                            name={icon}
                            size={16}
                            color={active ? theme.colors.onPrimary : theme.colors.onSurfaceMuted}
                          />
                          <Text style={[styles.modeChipText, active && styles.modeChipTextActive]}>{mode}</Text>
                        </Pressable>
                      );
                    })}
                  </View>
                </NeonCard>

                {selectedMode === 'buttons' ? (
                  <NeonCard style={styles.hudCard} tone="default">
                    <Text style={styles.cardTitle}>Choose your move</Text>
                    <Text style={styles.bodyText}>Play one round per move. Controls lock automatically on match completion.</Text>
                    <View style={styles.playGrid}>
                      {moveOptions.map((option) => (
                        <Pressable
                          key={option.move}
                          accessibilityRole="button"
                          accessibilityLabel={`Play ${option.label}`}
                          accessibilityState={{ disabled: matchLocked }}
                          onPress={() => playSelectedMove(option.move)}
                          disabled={matchLocked}
                          style={({ pressed }) => [
                            styles.moveTile,
                            pressed && !matchLocked && styles.pressed,
                            matchLocked && styles.muted,
                          ]}
                        >
                          <View style={styles.moveIconShell}>
                            <MaterialCommunityIcons name={option.icon} size={24} color={theme.colors.primary} />
                          </View>
                          <View style={styles.moveCopy}>
                            <Text style={styles.moveLabel}>{option.label}</Text>
                            <Text style={styles.moveCaption}>{option.caption}</Text>
                          </View>
                        </Pressable>
                      ))}
                    </View>
                  </NeonCard>
                ) : null}

                {selectedMode === 'vision' ? (
                  <NeonCard style={styles.hudCard}>
                    <Text style={styles.cardTitle}>Camera input</Text>
                    <Text style={styles.bodyText}>Capture one frame or import a still image. The app submits the detected move after inference.</Text>
                    <View style={styles.visionActionRow}>
                      <NeonButton label="Open camera" icon="camera-outline" onPress={openCameraCapture} disabled={matchLocked} />
                      <NeonButton label="Import image" icon="image-outline" variant="secondary" onPress={pickImageAndInfer} disabled={matchLocked} />
                    </View>

                    {visionPreviewUri ? (
                      <View style={styles.previewBox}>
                        <Image source={{ uri: visionPreviewUri }} style={styles.previewImage} contentFit="cover" />
                      </View>
                    ) : (
                      <View style={styles.emptyPanel}>
                        <MaterialCommunityIcons name="image-search-outline" size={30} color={theme.colors.primary} />
                        <Text style={styles.emptyTitle}>No frame analyzed yet</Text>
                        <Text style={styles.emptyText}>Use camera or image import to classify a move.</Text>
                      </View>
                    )}

                    {visionPrediction ? (
                      <View style={styles.visionResultPanel}>
                        <Text style={styles.statLabel}>Detected move</Text>
                        <Text style={styles.detectedMoveText}>{visionPrediction.move.toUpperCase()}</Text>
                        <Text style={styles.bodyText}>Confidence: {Math.round(visionPrediction.confidence * 100)}%</Text>
                        <Text style={styles.bodyText}>Inference path: {visionPrediction.source}</Text>
                        <NeonButton
                          label={matchLocked ? 'Match locked' : 'Submit detected move'}
                          icon="send-outline"
                          onPress={() => playSelectedMove(visionPrediction.move)}
                          disabled={matchLocked}
                        />
                      </View>
                    ) : null}

                    {visionError ? <Text style={styles.errorInlineText}>{visionError}</Text> : null}
                  </NeonCard>
                ) : null}

                {selectedMode === 'audio' ? (
                  <NeonCard style={styles.hudCard}>
                    <Text style={styles.cardTitle}>Voice input</Text>
                    <Text style={styles.bodyText}>Say rock, paper, scissors, or none. Speech maps directly to a server round.</Text>
                    <View style={styles.audioMicShell}>
                      <Pressable
                        accessibilityRole="button"
                        accessibilityLabel={recognizing ? 'Stop listening' : 'Start listening'}
                        accessibilityState={{ disabled: matchLocked, busy: recognizing }}
                        onPress={recognizing ? stopListening : startListening}
                        disabled={matchLocked}
                        style={({ pressed }) => [styles.audioMicButton, pressed && styles.pressed, matchLocked && styles.muted]}
                      >
                        <MaterialCommunityIcons name="microphone-outline" size={54} color={theme.colors.onPrimary} />
                      </Pressable>
                    </View>
                    <View style={styles.transcriptPanel}>
                      <Text style={styles.statLabel}>Transcript</Text>
                      <Text style={styles.transcriptText}>{transcript || 'Awaiting speech input...'}</Text>
                    </View>
                    <View style={styles.manualTranscriptRow}>
                      <TextInput
                        value={transcript}
                        onChangeText={setTranscript}
                        placeholder="Type a fallback transcript"
                        placeholderTextColor={theme.colors.onSurfaceMuted}
                        style={styles.input}
                      />
                    </View>
                    <View style={styles.audioMapPanel}>
                      <Text style={styles.statLabel}>Mapped move</Text>
                      <Text style={styles.detectedMoveText}>{transcriptMove ? transcriptMove.toUpperCase() : 'No match'}</Text>
                      <NeonButton
                        label={matchLocked ? 'Match locked' : 'Submit mapped move'}
                        icon="send-outline"
                        onPress={() => playSelectedMove((transcriptMove as GameMove) ?? 'none')}
                        disabled={matchLocked || !transcriptMove}
                      />
                    </View>
                    {audioError ? <Text style={styles.errorInlineText}>{audioError}</Text> : null}
                  </NeonCard>
                ) : null}

                <NeonCard style={styles.hudCard}>
                  <Text style={styles.cardTitle}>Latest round</Text>
                  {lastRound ? (
                    <View style={styles.roundResultBox}>
                      <View style={styles.detailRow}>
                        <Text style={styles.bodyText}>Round</Text>
                        <Text style={styles.bodyTextStrong}>{lastRound.round}</Text>
                      </View>
                      <View style={styles.detailRow}>
                        <Text style={styles.bodyText}>Player</Text>
                        <Text style={styles.bodyTextStrong}>{lastRound.player_move}</Text>
                      </View>
                      <View style={styles.detailRow}>
                        <Text style={styles.bodyText}>Server</Text>
                        <Text style={styles.bodyTextStrong}>{lastRound.server_move}</Text>
                      </View>
                      <View style={styles.detailRow}>
                        <Text style={styles.bodyText}>Winner</Text>
                        <Text style={styles.bodyTextStrong}>{lastRound.round_winner}</Text>
                      </View>
                      <View style={styles.detailRow}>
                        <Text style={styles.bodyText}>Score</Text>
                        <Text style={styles.bodyTextStrong}>{lastRound.player_score} - {lastRound.server_score}</Text>
                      </View>
                    </View>
                  ) : (
                    <View style={styles.emptyPanel}>
                      <MaterialCommunityIcons name="clock-outline" size={28} color={theme.colors.primary} />
                      <Text style={styles.emptyTitle}>Ready for round one</Text>
                      <Text style={styles.emptyText}>Choose a move to create the first entry.</Text>
                    </View>
                  )}

                  <NeonButton
                    label={busy ? 'Starting new match...' : 'New match'}
                    icon="refresh"
                    onPress={() => newMatchMutation.mutate()}
                    disabled={busy}
                  />
                </NeonCard>
              </>
            ) : (
              <NeonCard style={styles.summaryCard}>
                <Text style={styles.summaryTitle}>{matchSummaryTitle}</Text>
                <Text style={styles.summarySubtitle}>{matchSummarySubtitle}</Text>
                <View style={styles.summaryScoreRow}>
                  <View style={styles.summaryScoreCol}>
                    <Text style={styles.scoreLabel}>Player</Text>
                    <Text style={styles.scoreValue}>{lastRound?.player_score ?? 0}</Text>
                  </View>
                  <Text style={styles.versusText}>VS</Text>
                  <View style={styles.summaryScoreCol}>
                    <Text style={styles.scoreLabel}>House AI</Text>
                    <Text style={[styles.scoreValue, styles.houseScore]}>{lastRound?.server_score ?? 0}</Text>
                  </View>
                </View>
                <View style={styles.summaryHistoryRow}>
                  {roundHistory.slice(-5).map((round) => (
                    <View key={`${round.round}-${round.player_move}`} style={styles.summaryHistoryChip}>
                      <Text style={styles.summaryHistoryText}>R{round.round}</Text>
                      <Text style={styles.summaryHistoryText}>{round.player_move[0].toUpperCase()} vs {round.server_move[0].toUpperCase()}</Text>
                    </View>
                  ))}
                </View>
                <NeonButton
                  label={busy ? 'Starting new match...' : 'New match'}
                  icon="refresh"
                  onPress={() => newMatchMutation.mutate()}
                  disabled={busy}
                />
                <NeonButton label="View stats" icon="chart-box-outline" variant="secondary" onPress={() => setActiveTab('stats')} />
              </NeonCard>
            )}

            {loginError ? <Text style={styles.errorInlineText}>{loginError}</Text> : null}
          </ScrollView>
        ) : null}

        {activeTab === 'history' ? (
          <ScrollView contentContainerStyle={styles.tabContent}>
            <NeonCard style={styles.hudCard}>
              <Text style={styles.cardTitle}>Round history</Text>
              {roundHistory.length === 0 ? (
                <View style={styles.emptyPanel}>
                  <MaterialCommunityIcons name="timeline-clock-outline" size={30} color={theme.colors.primary} />
                  <Text style={styles.emptyTitle}>No rounds recorded</Text>
                  <Text style={styles.emptyText}>History appears as soon as a round completes.</Text>
                </View>
              ) : (
                <View style={styles.historyList}>
                  {roundHistory
                    .slice()
                    .reverse()
                    .map((round) => (
                      <View key={`${round.round}-${round.player_move}-${round.server_move}`} style={styles.historyRow}>
                        <Text style={styles.historyRound}>R{round.round}</Text>
                        <Text style={styles.historyMove}>{round.player_move.toUpperCase()}</Text>
                        <Text style={styles.historyVs}>vs</Text>
                        <Text style={styles.historyMove}>{round.server_move.toUpperCase()}</Text>
                        <Text style={styles.historyWinner}>{round.round_winner.toUpperCase()}</Text>
                      </View>
                    ))}
                </View>
              )}
            </NeonCard>
          </ScrollView>
        ) : null}

        {activeTab === 'stats' ? (
          <ScrollView contentContainerStyle={styles.tabContent}>
            <NeonCard style={styles.profileHeaderCard}>
              <View style={styles.profileTopRow}>
                <View style={styles.profileAvatarShell}>
                  <Text style={styles.profileAvatarInitial}>{(username?.[0] ?? 'P').toUpperCase()}</Text>
                </View>
                <View>
                  <Text style={styles.profileName}>{username}</Text>
                  <Text style={styles.profileRank}>Career profile</Text>
                </View>
              </View>

              <View style={styles.statsGrid}>
                <View style={styles.statCell}>
                  <Text style={styles.statLabel}>Win Rate</Text>
                  <Text style={styles.statValue}>{statsSnapshot.winRate.toFixed(1)}%</Text>
                </View>
                <View style={styles.statCell}>
                  <Text style={styles.statLabel}>Victories</Text>
                  <Text style={[styles.statValue, styles.houseScore]}>{statsSnapshot.wins}</Text>
                </View>
                <View style={styles.statCell}>
                  <Text style={styles.statLabel}>Rounds</Text>
                  <Text style={styles.statValue}>{statsSnapshot.rounds}</Text>
                </View>
              </View>
            </NeonCard>

            <NeonCard style={styles.hudCard}>
              <Text style={styles.cardTitle}>Match record</Text>
              <View style={styles.detailRow}>
                <Text style={styles.bodyText}>Sessions Started</Text>
                <Text style={styles.bodyTextStrong}>{statsSnapshot.sessionsStarted}</Text>
              </View>
              <View style={styles.detailRow}>
                <Text style={styles.bodyText}>Matches Completed</Text>
                <Text style={styles.bodyTextStrong}>{statsSnapshot.completed}</Text>
              </View>
              <View style={styles.detailRow}>
                <Text style={styles.bodyText}>Wins</Text>
                <Text style={styles.bodyTextStrong}>{statsSnapshot.wins}</Text>
              </View>
              <View style={styles.detailRow}>
                <Text style={styles.bodyText}>Losses</Text>
                <Text style={[styles.bodyTextStrong, styles.errorText]}>{statsSnapshot.losses}</Text>
              </View>
              <View style={styles.detailRow}>
                <Text style={styles.bodyText}>Draws</Text>
                <Text style={styles.bodyTextStrong}>{statsSnapshot.draws}</Text>
              </View>
            </NeonCard>
          </ScrollView>
        ) : null}

        {activeTab === 'settings' ? (
          <ScrollView contentContainerStyle={styles.tabContent}>
            <NeonCard style={styles.hudCard} tone="accent">
              <Text style={styles.cardTitle}>Account</Text>
              <View style={styles.detailRow}>
                <Text style={styles.bodyText}>Signed in as</Text>
                <Text style={styles.bodyTextStrong}>{username}</Text>
              </View>
              <View style={styles.detailRow}>
                <Text style={styles.bodyText}>Session ID</Text>
                <Text style={[styles.bodyTextStrong, styles.detailValue]} numberOfLines={1}>
                  {session?.sessionId ?? 'none'}
                </Text>
              </View>
              <NeonButton label="Sign out" icon="logout-variant" variant="secondary" onPress={handleLogout} />
            </NeonCard>

            <NeonCard style={styles.hudCard}>
              <Text style={styles.cardTitle}>Input mode</Text>
              <Text style={styles.bodyText}>Select the primary way to play each round.</Text>
              <View style={styles.modeRow}>
                {(['buttons', 'vision', 'audio'] as InputMode[]).map((mode) => {
                  const enabled = isModeReady(mode);
                  const active = selectedMode === mode;
                  const icon =
                    mode === 'buttons'
                      ? 'gesture-tap-button'
                      : mode === 'vision'
                        ? 'camera-iris'
                        : 'microphone-outline';
                  return (
                    <Pressable
                      key={`settings-${mode}`}
                      accessibilityRole="button"
                      accessibilityLabel={`Set ${mode} input mode${enabled ? '' : ', unavailable'}`}
                      accessibilityState={{ selected: active }}
                      onPress={() => handleModeSelection(mode)}
                      style={({ pressed }) => [
                        styles.modeChip,
                        active && styles.modeChipActive,
                        !enabled && styles.modeChipDisabled,
                        pressed && styles.pressed,
                      ]}
                    >
                      <MaterialCommunityIcons
                        name={icon}
                        size={16}
                        color={active ? theme.colors.onPrimary : theme.colors.onSurfaceMuted}
                      />
                      <Text style={[styles.modeChipText, active && styles.modeChipTextActive]}>{mode}</Text>
                    </Pressable>
                  );
                })}
              </View>
            </NeonCard>

            <NeonCard style={styles.hudCard}>
              <Text style={styles.cardTitle}>Model runtime</Text>
              {manifestQuery.isFetching ? (
                <Text style={styles.bodyText}>Refreshing manifest from server...</Text>
              ) : (
                <>
                  <View style={styles.detailRow}>
                    <Text style={styles.bodyText}>Enabled modes</Text>
                    <Text style={styles.bodyTextStrong}>{availableModes.join(', ')}</Text>
                  </View>
                  <View style={styles.detailRow}>
                    <Text style={styles.bodyText}>Vision slot</Text>
                    <Text style={styles.bodyTextStrong}>{manifest?.vision_model_slot ?? 'a'}</Text>
                  </View>
                  <View style={styles.detailRow}>
                    <Text style={styles.bodyText}>Vision model</Text>
                    <Text style={styles.bodyTextStrong}>
                      {visionModelAvailable ? manifest?.vision?.version ?? 'available' : 'not deployed'}
                    </Text>
                  </View>
                  <View style={styles.detailRow}>
                    <Text style={styles.bodyText}>Speech locale</Text>
                    <Text style={styles.bodyTextStrong}>{manifest?.audio?.browser_speech?.locale ?? 'en-US'}</Text>
                  </View>
                </>
              )}
              <NeonButton
                label="Refresh manifest"
                icon="sync"
                variant="secondary"
                onPress={() => void manifestQuery.refetch()}
                disabled={manifestQuery.isFetching}
              />
            </NeonCard>

            <NeonCard style={styles.hudCard}>
              <Text style={styles.cardTitle}>Preferences</Text>
              {MANUAL_API_ORIGIN_ENABLED ? (
                <View style={styles.endpointBlock}>
                  <Text style={styles.fieldLabel}>API Origin</Text>
                  <TextInput
                    autoCapitalize="none"
                    autoCorrect={false}
                    keyboardType="url"
                    value={apiBaseUrlDraft}
                    onChangeText={(value) => {
                      setApiBaseUrlDraft(value);
                      setApiBaseUrlError(null);
                    }}
                    style={styles.input}
                    placeholder="https://your-backend.example.com"
                    placeholderTextColor={theme.colors.onSurfaceMuted}
                  />
                  {apiBaseUrlError ? <Text style={styles.errorInlineText}>{apiBaseUrlError}</Text> : null}
                  <View style={styles.endpointActions}>
                    <NeonButton
                      label="Save origin"
                      icon="content-save-outline"
                      variant="secondary"
                      onPress={() => void commitApiBaseUrlDraft()}
                      style={styles.endpointAction}
                    />
                    <NeonButton
                      label="Use default"
                      icon="restore"
                      variant="quiet"
                      onPress={() => void handleResetApiBaseUrl()}
                      disabled={apiBaseUrl === DEFAULT_API_BASE_URL && apiBaseUrlDraft === DEFAULT_API_BASE_URL}
                      style={styles.endpointAction}
                    />
                  </View>
                </View>
              ) : null}
              <View style={styles.settingRow}>
                <View>
                  <Text style={styles.settingTitle}>Haptic feedback</Text>
                  <Text style={styles.bodyText}>Tactile response on action</Text>
                </View>
                <Switch
                  value={hapticsEnabled}
                  onValueChange={setHapticsEnabled}
                  trackColor={{ false: theme.colors.surfaceContainerHighest, true: theme.colors.primaryContainer }}
                  thumbColor={hapticsEnabled ? theme.colors.onSurface : theme.colors.onSurfaceMuted}
                />
              </View>
              <View style={styles.detailRow}>
                <Text style={styles.bodyText}>Session resume</Text>
                <Text style={styles.bodyTextStrong}>{session ? 'armed' : 'inactive'}</Text>
              </View>
              {MANUAL_API_ORIGIN_ENABLED ? (
                <View style={styles.detailRow}>
                  <Text style={styles.bodyText}>API origin</Text>
                  <Text style={[styles.bodyTextStrong, styles.detailValue]} numberOfLines={1}>
                    {apiBaseUrl}
                  </Text>
                </View>
              ) : null}
            </NeonCard>
          </ScrollView>
        ) : null}

        <BlurView intensity={55} tint="dark" style={styles.bottomNav}>
          {(
            [
              { key: 'arena', label: 'Arena', icon: 'gamepad-variant-outline' },
              { key: 'history', label: 'Rounds', icon: 'timeline-clock-outline' },
              { key: 'stats', label: 'Stats', icon: 'chart-box-outline' },
              { key: 'settings', label: 'System', icon: 'tune-variant' },
            ] as Array<{ key: PrimaryTab; label: string; icon: keyof typeof MaterialCommunityIcons.glyphMap }>
          ).map((item) => {
            const active = activeTab === item.key;
            return (
              <Pressable
                key={item.key}
                accessibilityRole="tab"
                accessibilityLabel={item.label}
                accessibilityState={{ selected: active }}
                onPress={() => setActiveTab(item.key)}
                style={({ pressed }) => [styles.navItem, active && styles.navItemActive, pressed && styles.pressed]}
              >
                <MaterialCommunityIcons name={item.icon} size={23} color={active ? theme.colors.primary : theme.colors.onSurfaceMuted} />
                <Text style={[styles.navLabel, active && styles.navLabelActive]}>{item.label}</Text>
              </Pressable>
            );
          })}
        </BlurView>

        {cameraOpen ? (
          <View style={styles.captureOverlay}>
            <CameraView ref={cameraRef} style={styles.captureCamera} facing="back" />
            <View style={styles.captureMaskTop} />
            <View style={styles.captureMaskBottom} />
            <View style={styles.captureHeader}>
              <Pressable
                accessibilityRole="button"
                accessibilityLabel="Close camera"
                onPress={() => setCameraOpen(false)}
                style={({ pressed }) => [styles.captureHeaderButton, pressed && styles.pressed]}
              >
                <MaterialCommunityIcons name="close" size={24} color={theme.colors.onSurface} />
              </Pressable>
              <Text style={styles.captureHeaderTitle}>Camera capture</Text>
              <Pressable
                accessibilityRole="button"
                accessibilityLabel="Flash setting"
                style={({ pressed }) => [styles.captureHeaderButton, pressed && styles.pressed]}
              >
                <MaterialCommunityIcons name="flash-outline" size={24} color={theme.colors.primary} />
              </Pressable>
            </View>
            <View style={styles.captureReticle}>
              <View style={styles.reticleCornerTopLeft} />
              <View style={styles.reticleCornerTopRight} />
              <View style={styles.reticleCornerBottomLeft} />
              <View style={styles.reticleCornerBottomRight} />
            </View>
            <View style={styles.captureFooter}>
              <Text style={styles.captureFooterStatus}>Ready to classify</Text>
              <Pressable
                accessibilityRole="button"
                accessibilityLabel="Capture frame"
                onPress={captureSingleFrame}
                style={({ pressed }) => [styles.captureButton, pressed && styles.pressed]}
              >
                <MaterialCommunityIcons name="camera-outline" size={34} color={theme.colors.onPrimary} />
              </Pressable>
            </View>
          </View>
        ) : null}
      </LinearGradient>
    </SafeAreaView>
  );
}

export default function App() {
  const [spaceLoaded] = useSpaceGrotesk({
    SpaceGrotesk_500Medium,
    SpaceGrotesk_700Bold,
  });
  const [manropeLoaded] = useManrope({
    Manrope_400Regular,
    Manrope_600SemiBold,
    Manrope_700Bold,
  });

  if (!spaceLoaded || !manropeLoaded) {
    return (
      <View style={styles.loadingScreen}>
        <StatusBar style="light" />
        <ActivityIndicator color={theme.colors.primary} size="large" />
      </View>
    );
  }

  return (
    <SafeAreaProvider>
      <QueryClientProvider client={queryClient}>
        <AppScreen />
      </QueryClientProvider>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  loadingScreen: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: theme.colors.surface,
    gap: theme.spacing.md,
  },
  loadingText: {
    color: theme.colors.onSurfaceMuted,
    fontFamily: 'Manrope_600SemiBold',
    fontSize: 14,
    letterSpacing: 0.8,
    textTransform: 'uppercase',
  },
  safe: {
    flex: 1,
    backgroundColor: theme.colors.surface,
  },
  appGradient: {
    flex: 1,
  },
  loginGradient: {
    flex: 1,
  },
  loginShell: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: theme.spacing.xl,
    gap: theme.spacing.md,
  },
  loginMarkRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: theme.spacing.lg,
  },
  appMark: {
    width: 54,
    height: 54,
    borderRadius: theme.radii.xl,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: theme.colors.primary,
    ...theme.shadows.soft,
  },
  loginStatusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    borderRadius: theme.radii.full,
    backgroundColor: theme.colors.neutralSoft,
    borderWidth: 1,
    borderColor: theme.colors.outlineVariant,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
  },
  loginStatusText: {
    color: theme.colors.onSurfaceMuted,
    fontFamily: 'Manrope_700Bold',
    fontSize: 12,
    letterSpacing: 0.2,
  },
  loginCard: {
    gap: theme.spacing.md,
    marginTop: theme.spacing.lg,
  },
  loginFootnote: {
    color: theme.colors.onSurfaceSubtle,
    fontFamily: 'Manrope_400Regular',
    fontSize: 12,
    lineHeight: 18,
  },
  topBar: {
    paddingHorizontal: theme.spacing.lg,
    paddingTop: theme.spacing.md,
    paddingBottom: theme.spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    overflow: 'hidden',
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.outlineVariant,
  },
  topBarTitleBlock: {
    alignItems: 'center',
    gap: 2,
  },
  topBarTitle: {
    color: theme.colors.onSurface,
    fontFamily: 'SpaceGrotesk_700Bold',
    fontSize: 24,
    letterSpacing: 0.2,
  },
  topBarSubTitle: {
    color: theme.colors.onSurfaceSubtle,
    fontFamily: 'Manrope_600SemiBold',
    fontSize: 11,
    letterSpacing: 0.2,
  },
  avatarChip: {
    width: 44,
    height: 44,
    borderRadius: theme.radii.lg,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: theme.colors.outlineStrong,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: theme.colors.surfaceContainer,
  },
  avatarInitial: {
    color: theme.colors.primary,
    fontFamily: 'SpaceGrotesk_700Bold',
    fontSize: 19,
  },
  settingsButton: {
    width: 44,
    height: 44,
    borderRadius: theme.radii.lg,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: theme.colors.neutralSoft,
    borderWidth: 1,
    borderColor: theme.colors.outlineVariant,
  },
  tabContent: {
    paddingHorizontal: theme.spacing.lg,
    paddingTop: theme.spacing.xl,
    paddingBottom: 132,
    gap: theme.spacing.md,
  },
  scoreCard: {
    backgroundColor: theme.colors.surfaceGlass,
  },
  scoreColumns: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  scoreCenterCol: {
    alignItems: 'center',
    gap: 6,
  },
  roundLabel: {
    color: theme.colors.secondary,
    fontFamily: 'Manrope_700Bold',
    fontSize: 12,
    letterSpacing: 0.4,
  },
  rightScore: {
    alignItems: 'flex-end',
  },
  scoreLabel: {
    color: theme.colors.onSurfaceMuted,
    fontFamily: 'Manrope_600SemiBold',
    fontSize: 11,
    letterSpacing: 0.6,
  },
  scoreValue: {
    color: theme.colors.onSurface,
    fontFamily: 'SpaceGrotesk_700Bold',
    fontSize: 58,
    lineHeight: 60,
  },
  houseScore: {
    color: theme.colors.secondary,
  },
  versusText: {
    color: theme.colors.onSurfaceSubtle,
    fontFamily: 'SpaceGrotesk_700Bold',
    fontSize: 24,
    lineHeight: 28,
  },
  hubCard: {
    gap: theme.spacing.md,
  },
  hubTitleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  liveChip: {
    borderRadius: theme.radii.full,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: 6,
    backgroundColor: theme.colors.primarySoft,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  liveDot: {
    width: 8,
    height: 8,
    borderRadius: 8,
    backgroundColor: theme.colors.success,
  },
  liveChipText: {
    color: theme.colors.primary,
    fontFamily: 'Manrope_600SemiBold',
    fontSize: 12,
    letterSpacing: 0.2,
  },
  cardEyebrow: {
    color: theme.colors.onSurfaceSubtle,
    fontFamily: 'Manrope_700Bold',
    fontSize: 11,
    letterSpacing: 0.7,
  },
  cardTitle: {
    color: theme.colors.onSurface,
    fontFamily: 'SpaceGrotesk_500Medium',
    fontSize: 24,
    lineHeight: 29,
    letterSpacing: 0,
  },
  title: {
    color: theme.colors.onSurface,
    fontFamily: 'SpaceGrotesk_700Bold',
    fontSize: 48,
    lineHeight: 52,
    letterSpacing: -0.3,
  },
  subtitle: {
    color: theme.colors.onSurfaceMuted,
    fontFamily: 'Manrope_400Regular',
    fontSize: 15,
    lineHeight: 23,
    marginBottom: theme.spacing.sm,
  },
  fieldLabel: {
    color: theme.colors.onSurfaceMuted,
    fontFamily: 'Manrope_600SemiBold',
    fontSize: 12,
    letterSpacing: 0.3,
  },
  input: {
    borderRadius: theme.radii.lg,
    backgroundColor: theme.colors.surface,
    borderWidth: 1,
    borderColor: theme.colors.outlineVariant,
    paddingHorizontal: theme.spacing.lg,
    paddingVertical: 13,
    color: theme.colors.onSurface,
    fontFamily: 'Manrope_400Regular',
    fontSize: 16,
  },
  hudCard: {
    gap: theme.spacing.md,
  },
  bodyText: {
    color: theme.colors.onSurfaceMuted,
    fontFamily: 'Manrope_400Regular',
    fontSize: 14,
    lineHeight: 21,
  },
  bodyTextStrong: {
    color: theme.colors.onSurface,
    fontFamily: 'Manrope_700Bold',
    fontSize: 15,
  },
  modeRow: {
    flexDirection: 'row',
    gap: theme.spacing.sm,
    marginBottom: theme.spacing.sm,
    flexWrap: 'wrap',
  },
  modeChip: {
    borderRadius: theme.radii.full,
    backgroundColor: theme.colors.neutralSoft,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: 10,
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    borderWidth: 1,
    borderColor: theme.colors.outlineVariant,
  },
  modeChipActive: {
    backgroundColor: theme.colors.primary,
    borderColor: 'transparent',
  },
  modeChipDisabled: {
    opacity: 0.45,
  },
  modeChipText: {
    color: theme.colors.onSurface,
    fontFamily: 'Manrope_700Bold',
    fontSize: 12,
    letterSpacing: 0.2,
  },
  modeChipTextActive: {
    color: theme.colors.onPrimary,
  },
  playGrid: {
    gap: theme.spacing.sm,
  },
  moveTile: {
    minHeight: 76,
    borderRadius: theme.radii.xl,
    backgroundColor: theme.colors.surfaceContainerLow,
    borderWidth: 1,
    borderColor: theme.colors.outlineVariant,
    padding: theme.spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.md,
  },
  moveIconShell: {
    width: 46,
    height: 46,
    borderRadius: theme.radii.lg,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: theme.colors.primarySoft,
  },
  moveCopy: {
    flex: 1,
    gap: 2,
  },
  moveLabel: {
    color: theme.colors.onSurface,
    fontFamily: 'Manrope_700Bold',
    fontSize: 17,
  },
  moveCaption: {
    color: theme.colors.onSurfaceSubtle,
    fontFamily: 'Manrope_400Regular',
    fontSize: 12,
  },
  visionActionRow: {
    gap: theme.spacing.sm,
    marginBottom: theme.spacing.sm,
  },
  previewBox: {
    borderRadius: theme.radii.xl,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: theme.colors.outlineStrong,
    backgroundColor: theme.colors.surfaceContainerHigh,
    height: 230,
    marginBottom: theme.spacing.sm,
  },
  previewImage: {
    width: '100%',
    height: '100%',
  },
  visionResultPanel: {
    borderRadius: theme.radii.xl,
    backgroundColor: theme.colors.surfaceContainer,
    padding: theme.spacing.lg,
    gap: theme.spacing.sm,
    borderWidth: 1,
    borderColor: theme.colors.outlineVariant,
  },
  emptyPanel: {
    borderRadius: theme.radii.xl,
    backgroundColor: theme.colors.neutralSoft,
    borderWidth: 1,
    borderColor: theme.colors.outlineVariant,
    padding: theme.spacing.lg,
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  emptyTitle: {
    color: theme.colors.onSurface,
    fontFamily: 'Manrope_700Bold',
    fontSize: 15,
  },
  emptyText: {
    color: theme.colors.onSurfaceMuted,
    fontFamily: 'Manrope_400Regular',
    fontSize: 13,
    lineHeight: 19,
    textAlign: 'center',
  },
  statLabel: {
    color: theme.colors.onSurfaceMuted,
    fontFamily: 'Manrope_600SemiBold',
    fontSize: 11,
    letterSpacing: 0.5,
  },
  detectedMoveText: {
    color: theme.colors.onSurface,
    fontFamily: 'SpaceGrotesk_700Bold',
    fontSize: 38,
    letterSpacing: 0,
    marginBottom: theme.spacing.xs,
  },
  audioMicShell: {
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: theme.spacing.sm,
  },
  audioMicButton: {
    width: 148,
    height: 148,
    borderRadius: 44,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: theme.colors.primary,
    ...theme.shadows.soft,
  },
  muted: {
    opacity: 0.45,
  },
  transcriptPanel: {
    borderRadius: theme.radii.xl,
    padding: theme.spacing.lg,
    backgroundColor: theme.colors.surfaceContainerLow,
    borderWidth: 1,
    borderColor: theme.colors.outlineVariant,
    marginBottom: theme.spacing.sm,
    minHeight: 92,
  },
  transcriptText: {
    color: theme.colors.onSurface,
    fontFamily: 'Manrope_600SemiBold',
    fontSize: 16,
  },
  manualTranscriptRow: {
    marginBottom: theme.spacing.sm,
  },
  audioMapPanel: {
    borderRadius: theme.radii.xl,
    backgroundColor: theme.colors.surfaceContainer,
    padding: theme.spacing.lg,
    gap: theme.spacing.sm,
    borderWidth: 1,
    borderColor: theme.colors.outlineVariant,
  },
  roundResultBox: {
    borderRadius: theme.radii.xl,
    backgroundColor: theme.colors.surfaceContainerLow,
    padding: theme.spacing.lg,
    gap: theme.spacing.xs,
    marginBottom: theme.spacing.md,
    borderWidth: 1,
    borderColor: theme.colors.outlineVariant,
  },
  summaryCard: {
    gap: theme.spacing.md,
  },
  summaryTitle: {
    color: theme.colors.primary,
    fontFamily: 'SpaceGrotesk_700Bold',
    fontSize: 48,
    lineHeight: 52,
    textAlign: 'center',
    letterSpacing: -0.2,
  },
  summarySubtitle: {
    color: theme.colors.onSurfaceMuted,
    fontFamily: 'Manrope_400Regular',
    fontSize: 15,
    lineHeight: 22,
    textAlign: 'center',
  },
  summaryScoreRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: theme.spacing.md,
  },
  summaryScoreCol: {
    alignItems: 'center',
    width: '40%',
  },
  summaryHistoryRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: theme.spacing.sm,
  },
  summaryHistoryChip: {
    borderRadius: theme.radii.lg,
    backgroundColor: theme.colors.surfaceContainerLow,
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: 8,
    minWidth: '31%',
    borderWidth: 1,
    borderColor: theme.colors.outlineVariant,
  },
  summaryHistoryText: {
    color: theme.colors.onSurface,
    fontFamily: 'Manrope_600SemiBold',
    fontSize: 11,
    letterSpacing: 0.7,
    textTransform: 'uppercase',
  },
  historyList: {
    gap: theme.spacing.sm,
  },
  historyRow: {
    borderRadius: theme.radii.lg,
    backgroundColor: theme.colors.surfaceContainerLow,
    padding: theme.spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
  },
  historyRound: {
    color: theme.colors.primary,
    fontFamily: 'SpaceGrotesk_700Bold',
    fontSize: 14,
    width: 38,
  },
  historyMove: {
    color: theme.colors.onSurface,
    fontFamily: 'Manrope_700Bold',
    fontSize: 12,
    width: 70,
    textAlign: 'center',
  },
  historyVs: {
    color: theme.colors.onSurfaceSubtle,
    fontFamily: 'SpaceGrotesk_700Bold',
    fontSize: 12,
  },
  historyWinner: {
    color: theme.colors.secondary,
    fontFamily: 'Manrope_700Bold',
    fontSize: 11,
    letterSpacing: 0.8,
    textTransform: 'uppercase',
    width: 85,
    textAlign: 'right',
  },
  profileHeaderCard: {
    gap: theme.spacing.md,
  },
  profileTopRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.md,
  },
  profileAvatarShell: {
    width: 76,
    height: 76,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: theme.colors.primary,
  },
  profileAvatarInitial: {
    color: theme.colors.onPrimary,
    fontFamily: 'SpaceGrotesk_700Bold',
    fontSize: 32,
  },
  profileName: {
    color: theme.colors.onSurface,
    fontFamily: 'SpaceGrotesk_700Bold',
    fontSize: 30,
    letterSpacing: -0.1,
  },
  profileRank: {
    color: theme.colors.primary,
    fontFamily: 'Manrope_600SemiBold',
    fontSize: 14,
    letterSpacing: 0.2,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: theme.spacing.sm,
  },
  statCell: {
    width: '31%',
    borderRadius: theme.radii.lg,
    backgroundColor: theme.colors.surfaceContainerLow,
    padding: theme.spacing.md,
    borderWidth: 1,
    borderColor: theme.colors.outlineVariant,
  },
  statValue: {
    color: theme.colors.onSurface,
    fontFamily: 'SpaceGrotesk_700Bold',
    fontSize: 34,
    lineHeight: 36,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 6,
    gap: theme.spacing.md,
  },
  detailValue: {
    flex: 1,
    textAlign: 'right',
  },
  endpointBlock: {
    gap: theme.spacing.sm,
  },
  endpointActions: {
    flexDirection: 'row',
    gap: theme.spacing.sm,
  },
  endpointAction: {
    flex: 1,
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    gap: theme.spacing.md,
  },
  settingTitle: {
    color: theme.colors.onSurface,
    fontFamily: 'Manrope_600SemiBold',
    fontSize: 16,
  },
  bottomNav: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingHorizontal: theme.spacing.md,
    paddingTop: theme.spacing.sm,
    paddingBottom: Platform.select({ ios: 22, android: 14, default: 14 }),
    borderTopLeftRadius: theme.radii.xxl,
    borderTopRightRadius: theme.radii.xxl,
    borderTopWidth: 1,
    borderTopColor: theme.colors.outlineStrong,
    backgroundColor: theme.colors.surfaceGlass,
    overflow: 'hidden',
  },
  navItem: {
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderRadius: theme.radii.lg,
    minWidth: 78,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  navItemActive: {
    backgroundColor: theme.colors.primarySoft,
    borderColor: theme.colors.outlineStrong,
  },
  navLabel: {
    color: theme.colors.onSurfaceMuted,
    fontFamily: 'Manrope_700Bold',
    fontSize: 10,
    letterSpacing: 0.2,
  },
  navLabelActive: {
    color: theme.colors.primary,
  },
  captureOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: theme.colors.surface,
    zIndex: 12,
  },
  captureCamera: {
    ...StyleSheet.absoluteFillObject,
  },
  captureMaskTop: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    height: 130,
    backgroundColor: theme.colors.surfaceScrim,
  },
  captureMaskBottom: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    height: 220,
    backgroundColor: theme.colors.surfaceScrim,
  },
  captureHeader: {
    position: 'absolute',
    left: theme.spacing.lg,
    right: theme.spacing.lg,
    top: 16,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  captureHeaderButton: {
    width: 40,
    height: 40,
    borderRadius: theme.radii.lg,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: theme.colors.surfaceGlass,
    borderWidth: 1,
    borderColor: theme.colors.outlineVariant,
  },
  captureHeaderTitle: {
    color: theme.colors.onSurface,
    fontFamily: 'SpaceGrotesk_700Bold',
    fontSize: 18,
    letterSpacing: 0,
  },
  captureReticle: {
    position: 'absolute',
    left: '16%',
    right: '16%',
    top: '28%',
    height: 320,
  },
  reticleCornerTopLeft: {
    position: 'absolute',
    left: 0,
    top: 0,
    width: 42,
    height: 42,
    borderLeftWidth: 2,
    borderTopWidth: 2,
    borderColor: theme.colors.primary,
  },
  reticleCornerTopRight: {
    position: 'absolute',
    right: 0,
    top: 0,
    width: 42,
    height: 42,
    borderRightWidth: 2,
    borderTopWidth: 2,
    borderColor: theme.colors.primary,
  },
  reticleCornerBottomLeft: {
    position: 'absolute',
    left: 0,
    bottom: 0,
    width: 42,
    height: 42,
    borderLeftWidth: 2,
    borderBottomWidth: 2,
    borderColor: theme.colors.primary,
  },
  reticleCornerBottomRight: {
    position: 'absolute',
    right: 0,
    bottom: 0,
    width: 42,
    height: 42,
    borderRightWidth: 2,
    borderBottomWidth: 2,
    borderColor: theme.colors.primary,
  },
  captureFooter: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 52,
    alignItems: 'center',
    gap: theme.spacing.md,
  },
  captureFooterStatus: {
    color: theme.colors.onSurface,
    fontFamily: 'Manrope_700Bold',
    fontSize: 15,
    letterSpacing: 0.2,
  },
  captureButton: {
    width: 98,
    height: 98,
    borderRadius: theme.radii.xxl,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: theme.colors.primary,
    ...theme.shadows.soft,
  },
  errorRoot: {
    flex: 1,
    paddingHorizontal: theme.spacing.lg,
    alignItems: 'center',
    justifyContent: 'center',
  },
  errorCard: {
    width: '100%',
    borderRadius: theme.radii.lg,
    backgroundColor: theme.colors.surfaceElevated,
    padding: theme.spacing.lg,
    alignItems: 'center',
    gap: theme.spacing.md,
    borderWidth: 1,
    borderColor: theme.colors.outlineStrong,
    ...theme.shadows.soft,
  },
  errorBadgeShell: {
    width: 110,
    height: 110,
    borderRadius: theme.radii.xxl,
    backgroundColor: theme.colors.errorSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  errorHeading: {
    color: theme.colors.error,
    fontFamily: 'SpaceGrotesk_700Bold',
    fontSize: 34,
    lineHeight: 39,
    textAlign: 'center',
    letterSpacing: 0,
  },
  errorSubheading: {
    color: theme.colors.onSurfaceMuted,
    fontFamily: 'Manrope_400Regular',
    fontSize: 15,
    lineHeight: 22,
    textAlign: 'center',
  },
  errorCode: {
    color: theme.colors.onSurfaceSubtle,
    fontFamily: 'Manrope_600SemiBold',
    fontSize: 12,
    letterSpacing: 0.2,
  },
  errorInlineText: {
    color: theme.colors.error,
    fontFamily: 'Manrope_600SemiBold',
    fontSize: 13,
  },
  errorText: {
    color: theme.colors.error,
  },
  pressed: {
    transform: [{ scale: 0.98 }],
  },
});
