import "react-native-url-polyfill/auto";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { createClient } from "@supabase/supabase-js";
import * as ImagePicker from "expo-image-picker";
import { StatusBar } from "expo-status-bar";
import { useVideoPlayer, VideoView } from "expo-video";
import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

const apiSetting = process.env.EXPO_PUBLIC_API_BASE_URL || "http://10.0.2.2:5000/api";
const apiBaseUrl = /^https?:\/\//i.test(apiSetting) ? apiSetting : `https://${apiSetting}`;
const normalizedApiUrl = apiBaseUrl.replace(/\/+$/, "").endsWith("/api")
  ? apiBaseUrl.replace(/\/+$/, "")
  : `${apiBaseUrl.replace(/\/+$/, "")}/api`;
const supabaseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL;
const supabaseKey = process.env.EXPO_PUBLIC_SUPABASE_PUBLISHABLE_KEY;
const supabase = supabaseUrl && supabaseKey
  ? createClient(supabaseUrl, supabaseKey, {
      auth: {
        storage: AsyncStorage,
        autoRefreshToken: true,
        persistSession: true,
        detectSessionInUrl: false,
      },
    })
  : null;

function ClipPlayer({ source, headers }) {
  const player = useVideoPlayer({ uri: source, headers: headers || {} });
  return <VideoView style={styles.video} player={player} nativeControls contentFit="contain" />;
}

function AuthScreen() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [registering, setRegistering] = useState(false);

  async function submit() {
    if (!supabase) {
      Alert.alert("Configuration needed", "Set the Supabase values in mobile/.env before signing in.");
      return;
    }
    if (!email.trim() || !password) {
      Alert.alert("Missing details", "Enter your email and password.");
      return;
    }
    setBusy(true);
    const action = registering
      ? supabase.auth.signUp({ email: email.trim(), password })
      : supabase.auth.signInWithPassword({ email: email.trim(), password });
    const { error } = await action;
    setBusy(false);
    if (error) Alert.alert("Could not sign in", error.message);
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="light" />
      <KeyboardAvoidingView style={styles.authWrap} behavior={Platform.OS === "ios" ? "padding" : undefined}>
        <View style={styles.brandMark}><Text style={styles.brandMarkText}>E</Text></View>
        <Text style={styles.eyebrow}>EVENT VIDEO INTELLIGENCE</Text>
        <Text style={styles.title}>EventLens</Text>
        <Text style={styles.subtitle}>Sign in to analyse an event recording.</Text>
        <TextInput autoCapitalize="none" keyboardType="email-address" placeholder="Email address" placeholderTextColor="#839087" value={email} onChangeText={setEmail} style={styles.input} />
        <TextInput secureTextEntry placeholder="Password" placeholderTextColor="#839087" value={password} onChangeText={setPassword} style={styles.input} />
        <Pressable onPress={submit} disabled={busy} style={styles.primaryButton}>
          {busy ? <ActivityIndicator color="#102016" /> : <Text style={styles.primaryButtonText}>{registering ? "Create account" : "Sign in"}</Text>}
        </Pressable>
        <Pressable onPress={() => setRegistering(!registering)} style={styles.linkButton}>
          <Text style={styles.linkText}>{registering ? "Already have an account? Sign in" : "New to EventLens? Create account"}</Text>
        </Pressable>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

export default function App() {
  const [session, setSession] = useState(null);
  const [authReady, setAuthReady] = useState(false);
  const [video, setVideo] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!supabase) {
      setAuthReady(true);
      return undefined;
    }
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session);
      setAuthReady(true);
    });
    const { data: listener } = supabase.auth.onAuthStateChange((_event, nextSession) => setSession(nextSession));
    return () => listener.subscription.unsubscribe();
  }, []);

  async function chooseVideo() {
    const selection = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["videos"],
      allowsEditing: false,
      quality: 1,
    });
    if (!selection.canceled) {
      setVideo(selection.assets[0]);
      setResult(null);
      setError("");
    }
  }

  async function analyzeVideo() {
    if (!video || !session?.access_token) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const form = new FormData();
      form.append("video", {
        uri: video.uri,
        name: video.fileName || "event-video.mp4",
        type: video.mimeType || "video/mp4",
      });
      const response = await fetch(`${normalizedApiUrl}/analyze`, {
        method: "POST",
        headers: { Authorization: `Bearer ${session.access_token}` },
        body: form,
      });
      const uploadData = await response.json();
      if (!response.ok) {
        throw new Error(uploadData.error || uploadData.message || "Video analysis failed.");
      }

      const jobId = uploadData.job_id || uploadData.report_id;
      if (!jobId) {
        throw new Error("The backend did not return an analysis job identifier.");
      }

      for (;;) {
        await new Promise((resolve) => setTimeout(resolve, 1500));
        const jobResponse = await fetch(
          `${normalizedApiUrl}/analyses/${encodeURIComponent(jobId)}`,
          { headers: { Authorization: `Bearer ${session.access_token}` } },
        );
        const job = await jobResponse.json();
        if (!jobResponse.ok) {
          throw new Error(job.error || job.message || "Could not retrieve analysis status.");
        }
        if (job.status === "failed") {
          throw new Error(job.error || "Video analysis failed.");
        }
        if (job.status === "completed") {
          setResult(job.result || job);
          return;
        }
      }
    } catch (requestError) {
      setError(requestError.message || "Could not reach the analysis API.");
    } finally {
      setBusy(false);
    }
  }

  async function signOut() {
    await supabase?.auth.signOut();
    setVideo(null);
    setResult(null);
  }

  if (!authReady) {
    return <SafeAreaView style={styles.loading}><ActivityIndicator color="#b9f36a" /></SafeAreaView>;
  }
  if (!session) return <AuthScreen />;

  const generatedUrl = result?.generated_video_url
    ? result.generated_video_url.startsWith("http")
      ? result.generated_video_url
      : `${normalizedApiUrl.replace(/\/api$/, "")}${result.generated_video_url}`
    : null;
  const eventClassification = result?.event_classification || {};
  const classificationScoreLabel =
    eventClassification.confidence_calibrated === true && result.confidence != null
      ? `${Math.round(result.confidence * 100)}% calibrated confidence`
      : eventClassification.model_score != null
        ? `Uncalibrated raw softmax score ${Number(eventClassification.model_score).toFixed(3)}`
        : null;
  const generatedHeaders = session?.access_token
    ? { Authorization: `Bearer ${session.access_token}` }
    : {};

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="dark" />
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.topbar}>
          <View><Text style={styles.eyebrow}>EVENT VIDEO INTELLIGENCE</Text><Text style={styles.heading}>Analysis</Text></View>
          <Pressable onPress={signOut}><Text style={styles.signOut}>Sign out</Text></Pressable>
        </View>

        <View style={styles.uploadPanel}>
          <Text style={styles.panelTitle}>Analyse a recording</Text>
          <Text style={styles.panelCopy}>Select an event video to detect objects and generate an event summary.</Text>
          <Pressable onPress={chooseVideo} disabled={busy} style={styles.secondaryButton}>
            <Text style={styles.secondaryButtonText}>{video?.fileName || "Choose a video"}</Text>
          </Pressable>
          {video && <ClipPlayer source={video.uri} />}
          <Pressable onPress={analyzeVideo} disabled={!video || busy} style={[styles.primaryButton, (!video || busy) && styles.disabledButton]}>
            {busy ? <ActivityIndicator color="#102016" /> : <Text style={styles.primaryButtonText}>Run event analysis</Text>}
          </Pressable>
          {error ? <Text style={styles.error}>{error}</Text> : null}
        </View>

        {result && (
          <View style={styles.results}>
            <Text style={styles.eyebrow}>ANALYSIS RESULT</Text>
            <Text style={styles.eventTitle}>{result.event_type || "Event detected"}</Text>
            {classificationScoreLabel && <Text style={styles.confidence}>{classificationScoreLabel}</Text>}
            <Text style={styles.summary}>{result.narrative || result.summary || "No summary was returned."}</Text>
            {result.model_quality && <Text style={styles.modelQuality}>Model status: {result.model_quality}</Text>}
            {Array.isArray(result.detections) && <Text style={styles.detectionCount}>{result.detections.length} object detections</Text>}
            {generatedUrl && <View style={styles.videoResult}><Text style={styles.panelTitle}>Generated video</Text><ClipPlayer source={generatedUrl} headers={generatedHeaders} /></View>}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: "#f5f8f2" },
  loading: { flex: 1, backgroundColor: "#102016", alignItems: "center", justifyContent: "center" },
  authWrap: { flex: 1, justifyContent: "center", padding: 28, backgroundColor: "#102016" },
  brandMark: { width: 48, height: 48, borderRadius: 14, backgroundColor: "#b9f36a", alignItems: "center", justifyContent: "center", marginBottom: 28 },
  brandMarkText: { color: "#102016", fontSize: 24, fontWeight: "800" },
  eyebrow: { color: "#779647", fontSize: 10, fontWeight: "800", letterSpacing: 1.6 },
  title: { color: "#f8fbf5", fontSize: 42, fontWeight: "800", marginTop: 8 },
  subtitle: { color: "#afbdaf", fontSize: 15, marginTop: 8, marginBottom: 28 },
  input: { height: 52, borderRadius: 10, paddingHorizontal: 15, backgroundColor: "#fff", color: "#17251b", marginBottom: 12, fontSize: 15 },
  primaryButton: { height: 52, borderRadius: 10, backgroundColor: "#b9f36a", alignItems: "center", justifyContent: "center", marginTop: 12 },
  primaryButtonText: { color: "#102016", fontSize: 15, fontWeight: "800" },
  linkButton: { alignItems: "center", padding: 18 },
  linkText: { color: "#d2f0ae", fontSize: 13, fontWeight: "600" },
  content: { padding: 20, paddingTop: 18, paddingBottom: 40 },
  topbar: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 22 },
  heading: { color: "#17251b", fontSize: 30, fontWeight: "800", marginTop: 5 },
  signOut: { color: "#657861", fontSize: 13, fontWeight: "700", padding: 8 },
  uploadPanel: { backgroundColor: "#fff", borderRadius: 14, padding: 18, borderWidth: 1, borderColor: "#e5ebdf" },
  panelTitle: { color: "#17251b", fontSize: 18, fontWeight: "800" },
  panelCopy: { color: "#788579", fontSize: 13, lineHeight: 20, marginTop: 7, marginBottom: 15 },
  secondaryButton: { minHeight: 48, borderRadius: 9, borderWidth: 1, borderColor: "#dbe4d4", padding: 12, justifyContent: "center" },
  secondaryButtonText: { color: "#4e6544", fontWeight: "700", fontSize: 13 },
  video: { width: "100%", height: 220, borderRadius: 10, backgroundColor: "#101810", marginTop: 14 },
  disabledButton: { opacity: 0.55 },
  error: { color: "#a33030", fontSize: 13, lineHeight: 19, marginTop: 12 },
  results: { marginTop: 24, padding: 18, backgroundColor: "#fff", borderRadius: 14, borderWidth: 1, borderColor: "#e5ebdf" },
  eventTitle: { color: "#17251b", fontSize: 26, fontWeight: "800", textTransform: "capitalize", marginTop: 8 },
  confidence: { color: "#6e963a", fontSize: 12, fontWeight: "800", marginTop: 5 },
  summary: { color: "#59675b", fontSize: 14, lineHeight: 22, marginTop: 13 },
  modelQuality: { color: "#879387", fontSize: 11, marginTop: 10 },
  detectionCount: { color: "#354838", fontSize: 12, fontWeight: "700", marginTop: 16, paddingTop: 14, borderTopWidth: 1, borderTopColor: "#edf1e9" },
  videoResult: { marginTop: 20 },
});
