//! `chimera texture …`: key- and tempo-locked layers for a track: pads, choirs, vocal chops,
//! drones and songbirds, each a seamless stereo loop of whole bars plus its voices as stems.
//!
//! Everything is placed on a beat grid at `--bpm` and on the notes of `--key` (tuned by `--tune`
//! cents, for tracks that are not at A440). A voice is a separate `HumanVoice` (or songbird) with
//! its own body between male, female and child, so a stack sounds like several people rather
//! than one voice copied. Renders run one voice per thread.
//!
//! Loops: each voice renders with a pre-roll (onset consonants come before the beat) and a
//! release tail; both are wrapped around the loop, so it plays back seamlessly at any length.

use crate::js::fixed;
use crate::render::g2p;
use crate::{Ctx, FS, peak};
use chimera::analysis::pitch::{YinOpts, yin_pitch};
use chimera::anatomy::human::BodyMix;
use chimera::creatures::creature_by_id;
use chimera::creatures::voice::CreatureVoice;
use chimera::phonetics::Syllable;
use chimera::phonetics::sing::{SongOptions, SungNote, midi_to_hz, plan_song};
use chimera::phonetics::syllables::lyrics_to_syllables;
use chimera::voice::human_voice::{HumanVoice, SourcePatch};

pub const USAGE: &str = "\
  texture <kind> [outdir] --key <key> [--bpm 124] [--bars 8] [--tune 0] [--seed 1]
          [--chords \"i VI III VII\"] [--chord-bars 2] [--voices n] [--vowels \"ah\"]
          [--lyrics \"la lu\"] [--breath 0.3] [--vibrato cents] [--octave 0]
                 kinds: pad (held chords, one vowel), choir (sung chords), chops (16th-note
                 syllable patterns on the chords), drone (root and fifth, vowels turning),
                 birds (songbirds singing the chord tones). --key: \"A minor\", Am, F#, Bbmaj, 8A.
                 Writes <outdir>/<kind>-<key>-<bpm>bpm.wav (a stereo loop of whole bars), one
                 stem per voice (they sum to the loop) and a .json of every note.";

const SCALE_MAJOR: [i32; 7] = [0, 2, 4, 5, 7, 9, 11];
const SCALE_MINOR: [i32; 7] = [0, 2, 3, 5, 7, 8, 10];
const NOTE_NAMES: [&str; 12] = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
// Camelot numbers by tonic pitch class (as flow-analysis's key_detect)
const CAMELOT_MAJOR: [u32; 12] = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1];
const CAMELOT_MINOR: [u32; 12] = [5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10];
/// seconds rendered before the loop's first beat (onset consonants, breath)
const PRE_ROLL: f64 = 1.0;
/// seconds rendered after its last beat (releases)
const TAIL: f64 = 2.5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Key {
    pub tonic: i32,
    pub minor: bool,
}

impl Key {
    pub fn scale(&self) -> &'static [i32; 7] {
        if self.minor { &SCALE_MINOR } else { &SCALE_MAJOR }
    }

    /// pitch class of scale degree `d` (any integer; wraps by octaves)
    pub fn degree_semis(&self, d: i32) -> i32 {
        let s = self.scale();
        let (oct, i) = (d.div_euclid(7), d.rem_euclid(7));
        self.tonic + s[i as usize] + 12 * oct
    }

    /// "Am", "F#", for file names
    pub fn short(&self) -> String {
        format!("{}{}", NOTE_NAMES[self.tonic as usize], if self.minor { "m" } else { "" })
    }

    pub fn camelot(&self) -> String {
        let t = self.tonic as usize;
        if self.minor { format!("{}A", CAMELOT_MINOR[t]) } else { format!("{}B", CAMELOT_MAJOR[t]) }
    }
}

/// "A minor", "Am", "F#m", "Bbmaj", "C", "c minor", or Camelot "8A" / "11B".
pub fn parse_key(s: &str) -> Result<Key, String> {
    let t = s.trim();
    let err = || format!("can't read key {s:?}; try \"A minor\", Am, F#, Bbmaj or Camelot 8A");
    let lower = t.to_lowercase();
    if let Some(letter) = lower.chars().last().filter(|c| *c == 'a' || *c == 'b')
        && let Ok(n) = lower[..lower.len() - 1].parse::<u32>()
    {
        let table = if letter == 'a' { &CAMELOT_MINOR } else { &CAMELOT_MAJOR };
        let tonic = table.iter().position(|&x| x == n).ok_or_else(err)?;
        return Ok(Key { tonic: tonic as i32, minor: letter == 'a' });
    }
    let mut chars = t.chars();
    let letter = chars.next().ok_or_else(err)?.to_ascii_uppercase();
    let mut pc = match letter {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => return Err(err()),
    };
    let mut rest: String = chars.collect();
    if rest.starts_with('#') || rest.starts_with('♯') {
        pc += 1;
        rest = rest.chars().skip(1).collect();
    } else if rest.starts_with('b') || rest.starts_with('♭') {
        pc -= 1;
        rest = rest.chars().skip(1).collect();
    }
    let mode = rest.trim().to_lowercase();
    let minor = match mode.as_str() {
        "" | "maj" | "major" | "M" => false,
        "m" | "min" | "minor" | "-" => true,
        _ if rest.trim() == "M" => false,
        _ => return Err(err()),
    };
    Ok(Key { tonic: (pc + 12) % 12, minor })
}

/// A chord: scale degrees (0-based from the tonic) stacked in thirds.
#[derive(Clone, Debug, PartialEq)]
pub struct Chord {
    pub name: String,
    pub degrees: Vec<i32>,
}

/// Roman numerals in the key ("i VI III VII", "I V vi IV", "ii7"); case is ignored, the key
/// decides each chord's quality. "7" adds the seventh, "sus2"/"sus4" replace the third, "5" is
/// root and fifth only.
pub fn parse_chords(s: &str) -> Result<Vec<Chord>, String> {
    let mut out = Vec::new();
    for tok in s.split_whitespace() {
        let lower = tok.to_lowercase();
        let numeral_len = lower.chars().take_while(|c| *c == 'i' || *c == 'v').count();
        let root = match &lower[..numeral_len] {
            "i" => 0,
            "ii" => 1,
            "iii" => 2,
            "iv" => 3,
            "v" => 4,
            "vi" => 5,
            "vii" => 6,
            _ => return Err(format!("can't read chord {tok:?}; use roman numerals like i VI III VII")),
        };
        let degrees = match &lower[numeral_len..] {
            "" => vec![root, root + 2, root + 4],
            "7" => vec![root, root + 2, root + 4, root + 6],
            "9" | "add9" => vec![root, root + 2, root + 4, root + 8],
            "sus2" => vec![root, root + 1, root + 4],
            "sus4" => vec![root, root + 3, root + 4],
            "5" => vec![root, root + 4],
            other => return Err(format!("chord {tok:?}: unknown suffix {other:?} (7, 9, sus2, sus4, 5)")),
        };
        out.push(Chord { name: tok.to_string(), degrees });
    }
    if out.is_empty() {
        return Err("no chords".into());
    }
    Ok(out)
}

/// mulberry32: small, seeded, the same on every machine
pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u32) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9).wrapping_add(0x6D2B_79F5))
    }

    pub fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x6D2B_79F5);
        let mut t = self.0;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        ((t ^ (t >> 14)) as f64) / 4_294_967_296.0
    }

    pub fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next()
    }

    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[((self.next() * xs.len() as f64) as usize).min(xs.len() - 1)]
    }
}

/// One sung event: a note (midi, fractional for tuning) or a rest (`midi` None), in beats.
#[derive(Clone, Debug)]
pub struct Event {
    pub midi: Option<f64>,
    pub beats: f64,
    pub syl: Option<Syllable>,
}

#[derive(Clone, Debug)]
pub struct HumanPart {
    pub name: String,
    pub body: BodyMix,
    pub pan: f64,
    pub gain: f64,
    pub source: SourcePatch,
    pub vibrato: f64,
    /// beat offset of the first event from the loop start (humanised entries, negative = early)
    pub offset_beats: f64,
    pub events: Vec<Event>,
}

#[derive(Clone, Debug)]
pub struct BirdNote {
    pub beat: f64,
    pub beats: f64,
    pub midi: f64,
    /// right voice interval in semitones (diatonic in the key)
    pub interval: f64,
}

#[derive(Clone, Debug)]
pub struct BirdPart {
    pub name: String,
    pub pan: f64,
    pub gain: f64,
    pub notes: Vec<BirdNote>,
}

pub enum Part {
    Human(HumanPart),
    Bird(BirdPart),
}

impl Part {
    fn name(&self) -> &str {
        match self {
            Part::Human(p) => &p.name,
            Part::Bird(p) => &p.name,
        }
    }

    fn pan_gain(&self) -> (f64, f64) {
        match self {
            Part::Human(p) => (p.pan, p.gain),
            Part::Bird(p) => (p.pan, p.gain),
        }
    }
}

pub struct Settings {
    pub key: Key,
    pub bpm: f64,
    pub bars: usize,
    pub tune_cents: f64,
    pub seed: u32,
    pub chords: Vec<Chord>,
    pub chord_bars: usize,
    pub voices: Option<usize>,
    pub vowels: Vec<String>,
    pub lyrics: Vec<String>,
    pub breath: f64,
    pub vibrato: Option<f64>,
    pub octave: i32,
}

impl Settings {
    fn beats(&self) -> f64 {
        self.bars as f64 * 4.0
    }

    /// the chord sounding in bar `bar` (0-based)
    fn chord_at_bar(&self, bar: usize) -> &Chord {
        &self.chords[(bar / self.chord_bars.max(1)) % self.chords.len()]
    }

    /// the chords in loop order with their lengths in beats (repeating to fill the bars)
    fn chord_spans(&self) -> Vec<(&Chord, f64)> {
        let mut spans: Vec<(&Chord, f64)> = Vec::new();
        for bar in 0..self.bars {
            let c = self.chord_at_bar(bar);
            match spans.last_mut() {
                Some((last, b)) if std::ptr::eq(*last, c) && bar % self.chord_bars.max(1) != 0 => *b += 4.0,
                _ => spans.push((c, 4.0)),
            }
        }
        spans
    }

    /// midi notes of the chord's tones near `center` in every octave (tuning applied later)
    fn chord_midis(&self, c: &Chord, lo: i32, hi: i32) -> Vec<i32> {
        let pcs: Vec<i32> = c.degrees.iter().map(|&d| self.key.degree_semis(d).rem_euclid(12)).collect();
        (lo..=hi).filter(|m| pcs.contains(&m.rem_euclid(12))).collect()
    }

    fn root_pc(&self, c: &Chord) -> i32 {
        self.key.degree_semis(c.degrees[0]).rem_euclid(12)
    }

    fn tuned(&self, midi: i32) -> f64 {
        midi as f64 + self.tune_cents / 100.0
    }
}

/// a voice's range and body, from bass (0) to top (1)
fn voice_layout(t: f64, rng: &mut Rng) -> (f64, BodyMix) {
    let center = 45.0 + 27.0 * t; // A2 .. C5
    let female = ((t - 0.3) / 0.4).clamp(0.0, 1.0);
    let child = ((t - 0.75) / 0.25).clamp(0.0, 1.0) * 0.6;
    let jitter = 0.12;
    (
        center,
        BodyMix::new(female + rng.range(-jitter, jitter), child + rng.range(-jitter * 0.5, jitter * 0.5)),
    )
}

/// Voice-led chord tones, one line per voice: the bass sings the roots; each upper voice moves to
/// the nearest chord tone, pulled toward its own register and away from tones already taken.
fn voice_lines(s: &Settings, n: usize, centers: &[f64]) -> Vec<Vec<(i32, f64)>> {
    let spans = s.chord_spans();
    let mut lines: Vec<Vec<(i32, f64)>> = vec![Vec::new(); n];
    let mut prev: Vec<Option<i32>> = vec![None; n];
    for (chord, beats) in spans {
        let mut taken: Vec<i32> = Vec::new();
        for v in 0..n {
            let c = centers[v];
            let pool = s.chord_midis(chord, (c - 9.0) as i32, (c + 9.0) as i32);
            let root = s.root_pc(chord);
            let best = pool
                .iter()
                .copied()
                .filter(|m| v != 0 || m.rem_euclid(12) == root)
                .min_by(|&a, &b| {
                    let cost = |m: i32| {
                        let lead = prev[v].map_or(0.0, |p| (m - p).abs() as f64);
                        let reg = 0.35 * (m as f64 - c).abs();
                        let dup = if taken.contains(&m.rem_euclid(12)) { 2.5 } else { 0.0 };
                        lead + reg + dup
                    };
                    cost(a).partial_cmp(&cost(b)).unwrap()
                })
                .unwrap_or(c as i32);
            taken.push(best.rem_euclid(12));
            prev[v] = Some(best);
            lines[v].push((best, beats));
        }
    }
    lines
}

fn syllable(g: &chimera::phonetics::english::EnglishG2P, word: &str) -> Result<Syllable, String> {
    lyrics_to_syllables(word, g)
        .into_iter()
        .next()
        .ok_or_else(|| format!("no syllable in {word:?}"))
}

/// the same vowel carried on to a new pitch, without re-articulating
fn melisma(s: &Syllable) -> Syllable {
    let mut m = s.clone();
    m.onset.clear();
    m.coda.clear();
    m.melisma = Some(true);
    m
}

fn soft_source(breath: f64, rng: &mut Rng) -> SourcePatch {
    SourcePatch {
        pressure: Some(rng.range(6.5, 8.0)),
        breathiness: Some((breath + rng.range(-0.05, 0.05)).clamp(0.0, 1.0)),
        tension: Some(rng.range(0.45, 0.55)),
        drift: Some(rng.range(1.5, 3.0)),
        brightness: Some(rng.range(0.15, 0.3)),
        ..Default::default()
    }
}

fn spread_pan(v: usize, n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    // alternate sides, widening with height; the bass stays near the centre
    let side = if v % 2 == 0 { -1.0 } else { 1.0 };
    side * (0.15 + 0.6 * v as f64 / (n - 1) as f64)
}

/// pad and choir: voice-led chords; pad holds one vowel through every change, choir sings a
/// syllable per chord with a breath before the next
fn chordal(s: &Settings, choir: bool) -> Result<Vec<Part>, String> {
    let g = g2p()?;
    let n = s.voices.unwrap_or(if choir { 8 } else { 6 });
    let mut rng = Rng::new(s.seed);
    let layout: Vec<(f64, BodyMix)> = (0..n)
        .map(|v| voice_layout(if n > 1 { v as f64 / (n - 1) as f64 } else { 0.5 }, &mut rng))
        .collect();
    let centers: Vec<f64> = layout.iter().map(|l| l.0 + 12.0 * s.octave as f64).collect();
    let lines = voice_lines(s, n, &centers);
    let words: Vec<String> = if choir && !s.lyrics.is_empty() { s.lyrics.clone() } else { s.vowels.clone() };
    let syls: Vec<Syllable> = words.iter().map(|w| syllable(&g, w)).collect::<Result<_, _>>()?;
    let mut parts = Vec::new();
    for (v, line) in lines.iter().enumerate() {
        let mut events: Vec<Event> = Vec::new();
        let mut last_syl: Option<Syllable> = None;
        for (k, &(m, beats)) in line.iter().enumerate() {
            let word = &syls[k % syls.len()];
            if choir {
                // breathe for the last eighth of each chord (staggered: odd voices breathe late)
                let breath = if v % 2 == 0 { 0.5 } else { 0.25 };
                events.push(Event { midi: Some(s.tuned(m)), beats: beats - breath, syl: Some(word.clone()) });
                events.push(Event { midi: None, beats: breath, syl: None });
            } else {
                // a held line: same pitch extends, a new pitch carries the vowel (or turns to the next)
                let same_vowel = last_syl.as_ref().is_some_and(|l| l.nucleus.ipa == word.nucleus.ipa);
                if let Some(e) = events.last_mut().filter(|e| e.midi == Some(s.tuned(m)) && same_vowel) {
                    e.beats += beats;
                    continue;
                }
                let syl = if same_vowel { melisma(word) } else { word.clone() };
                events.push(Event { midi: Some(s.tuned(m)), beats, syl: Some(syl) });
            }
            last_syl = Some(word.clone());
        }
        let (_, body) = layout[v];
        let mut source = soft_source(if choir { s.breath * 0.6 } else { s.breath }, &mut rng);
        if v == 0 && !choir {
            source.roughness = Some(0.25); // grit in the bass (not fry: fry flattens the pitch)
        }
        parts.push(Part::Human(HumanPart {
            name: format!("voice-{}-{}", v + 1, register_name(v, n)),
            body,
            pan: spread_pan(v, n),
            gain: if v == 0 { 0.9 } else { 0.75 },
            source,
            vibrato: s.vibrato.unwrap_or(if choir { 28.0 } else { 6.0 }) * rng.range(0.7, 1.2),
            offset_beats: if choir { rng.range(-0.06, 0.04) } else { rng.range(-0.15, 0.0) },
            events,
        }));
    }
    Ok(parts)
}

fn register_name(v: usize, n: usize) -> &'static str {
    let t = if n > 1 { v as f64 / (n - 1) as f64 } else { 0.5 };
    match t {
        t if t < 0.2 => "bass",
        t if t < 0.45 => "tenor",
        t if t < 0.75 => "alto",
        _ => "soprano",
    }
}

/// drone: root, fifth and octave held through the loop, each voice turning its vowel per bar
fn drone(s: &Settings) -> Result<Vec<Part>, String> {
    let g = g2p()?;
    let mut rng = Rng::new(s.seed);
    let vowels: Vec<String> = if s.vowels.len() > 1 {
        s.vowels.clone()
    } else {
        ["ooh", "oh", "ah", "oh"].iter().map(|x| x.to_string()).collect()
    };
    let syls: Vec<Syllable> = vowels.iter().map(|w| syllable(&g, w)).collect::<Result<_, _>>()?;
    let base = 36 + s.key.tonic + 12 * s.octave; // C2..B2
    // (interval above the root, body, register name, roughness). No fry: it slackens the folds
    // and drags the note flat (a quarter of fry put an A2 2.6 semitones low).
    let voices: Vec<(i32, BodyMix, &str, f64)> = vec![
        (0, BodyMix::new(0.0, 0.0), "root", 0.35),
        (7, BodyMix::new(0.1, 0.0), "fifth", 0.2),
        (12, BodyMix::new(0.45, 0.0), "octave", 0.0),
        (19, BodyMix::new(0.9, 0.3), "high-fifth", 0.0),
    ];
    let n = s.voices.unwrap_or(4).min(voices.len());
    let mut parts = Vec::new();
    for (v, &(iv, body, label, rough)) in voices.iter().take(n).enumerate() {
        // each voice turns its vowels at its own pace (1, 2 or 3 bars), so they shift out of step
        let per = 1 + v % 3;
        let mut events: Vec<Event> = Vec::new();
        let mut bar = 0;
        let mut k = v;
        while bar < s.bars {
            let len = per.min(s.bars - bar);
            events.push(Event {
                midi: Some(s.tuned(base + iv)),
                beats: len as f64 * 4.0,
                syl: Some(syls[k % syls.len()].clone()),
            });
            bar += len;
            k += 1;
        }
        let mut source = soft_source(s.breath, &mut rng);
        source.roughness = Some(rough.max(0.1));
        parts.push(Part::Human(HumanPart {
            name: format!("voice-{}-{label}", v + 1),
            body: BodyMix::new(body.female + rng.range(-0.05, 0.05), body.child),
            pan: spread_pan(v, n),
            gain: [1.0, 0.7, 0.6, 0.45][v],
            source,
            vibrato: s.vibrato.unwrap_or(4.0),
            offset_beats: rng.range(-0.2, 0.0),
            events,
        }));
    }
    Ok(parts)
}

/// chops: each voice a seeded one- or two-bar pattern of 16th-note syllables on the chord tones,
/// repeated over the loop and following the chords
fn chops(s: &Settings) -> Result<Vec<Part>, String> {
    let g = g2p()?;
    let words: Vec<String> = if s.lyrics.is_empty() {
        ["da", "ya", "na", "oh", "ba", "ee", "dee", "la"].iter().map(|x| x.to_string()).collect()
    } else {
        s.lyrics.clone()
    };
    let syls: Vec<Syllable> = words.iter().map(|w| syllable(&g, w)).collect::<Result<_, _>>()?;
    let n = s.voices.unwrap_or(2);
    let mut parts = Vec::new();
    for v in 0..n {
        let mut rng = Rng::new(s.seed.wrapping_mul(31).wrapping_add(v as u32));
        let center = 60.0 + 7.0 * v as f64 + 12.0 * s.octave as f64;
        let body = if v % 2 == 0 { BodyMix::new(rng.range(0.7, 1.0), 0.0) } else { BodyMix::new(0.5, rng.range(0.3, 0.7)) };
        // the pattern: (step, length in steps, chord-tone index, syllable index), steps of a 16th
        let pattern_bars = if rng.next() < 0.5 { 1 } else { 2 };
        let steps = 16 * pattern_bars;
        let mut pattern: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut step = 0;
        while step < steps {
            let density = if step % 4 == 0 { 0.7 } else { 0.45 };
            if rng.next() < density {
                let len = if rng.next() < 0.7 { 1 } else { 2 }.min(steps - step);
                pattern.push((step, len, (rng.next() * 4.0) as usize, (rng.next() * syls.len() as f64) as usize));
                step += len;
            } else {
                step += 1;
            }
        }
        // realise it bar by bar: chord tones nearest the voice's centre, the rests in between
        let mut events: Vec<Event> = Vec::new();
        let mut at = 0.0f64;
        for rep in 0..s.bars.div_ceil(pattern_bars) {
            for &(st, len, tone, si) in &pattern {
                let beat = (rep * steps + st) as f64 / 4.0;
                if beat >= s.beats() {
                    break;
                }
                let bar = (beat / 4.0) as usize;
                let mut tones = s.chord_midis(s.chord_at_bar(bar), (center - 6.0) as i32, (center + 7.0) as i32);
                tones.sort();
                let m = tones[tone % tones.len()];
                if beat > at {
                    events.push(Event { midi: None, beats: beat - at, syl: None });
                }
                let beats = len as f64 / 4.0;
                events.push(Event { midi: Some(s.tuned(m)), beats, syl: Some(syls[si].clone()) });
                at = beat + beats;
            }
        }
        if s.beats() > at {
            events.push(Event { midi: None, beats: s.beats() - at, syl: None });
        }
        parts.push(Part::Human(HumanPart {
            name: format!("voice-{}-chops", v + 1),
            body,
            pan: if n == 1 { 0.0 } else if v % 2 == 0 { -0.45 } else { 0.45 },
            gain: 0.8,
            source: SourcePatch {
                pressure: Some(rng.range(8.0, 10.0)),
                breathiness: Some(s.breath * 0.5),
                tension: Some(0.55),
                drift: Some(1.0),
                ..Default::default()
            },
            vibrato: s.vibrato.unwrap_or(0.0),
            offset_beats: 0.0,
            events,
        }));
    }
    Ok(parts)
}

/// birds: songbirds chirping and trilling the chord tones, their second voice a diatonic third
/// or sixth above
fn birds(s: &Settings) -> Result<Vec<Part>, String> {
    let n = s.voices.unwrap_or(2);
    let mut parts = Vec::new();
    for v in 0..n {
        let mut rng = Rng::new(s.seed.wrapping_mul(97).wrapping_add(v as u32));
        // songbird range: around 2-4.5 kHz sounds most like a bird; one octave band per bird
        let lo = 95 + 3 * (v as i32 % 3) + 12 * s.octave; // B6.. (1976 Hz)
        let mut notes: Vec<BirdNote> = Vec::new();
        let mut beat = rng.range(0.0, 2.0).floor() / 2.0;
        while beat < s.beats() {
            let bar = (beat / 4.0) as usize;
            let chord = s.chord_at_bar(bar);
            let tones = s.chord_midis(chord, lo, lo + 12);
            let r = rng.next();
            if r < 0.35 {
                // a trill: 32nds alternating with the scale step above
                let m = *rng.pick(&tones);
                let up = scale_step(s, m, 1);
                let count = 4 + (rng.next() * 6.0) as usize;
                for i in 0..count {
                    let b = beat + i as f64 / 8.0;
                    if b >= s.beats() {
                        break;
                    }
                    notes.push(BirdNote { beat: b, beats: 0.1, midi: s.tuned(if i % 2 == 0 { m } else { up }), interval: 0.0 });
                }
                beat += count as f64 / 8.0 + 0.5;
            } else if r < 0.8 {
                // a phrase of chirps walking the chord
                let mut m = *rng.pick(&tones);
                for _ in 0..(2 + (rng.next() * 3.0) as usize) {
                    if beat >= s.beats() {
                        break;
                    }
                    let third = (scale_step(s, m, 2) - m) as f64;
                    notes.push(BirdNote { beat, beats: 0.2, midi: s.tuned(m), interval: if rng.next() < 0.5 { third } else { 0.0 } });
                    beat += if rng.next() < 0.6 { 0.25 } else { 0.5 };
                    m = scale_step(s, m, if rng.next() < 0.5 { 1 } else { -1 });
                }
                beat += 0.5;
            } else {
                // a held whistle with a sixth above
                let m = *rng.pick(&tones);
                let sixth = (scale_step(s, m, 5) - m) as f64;
                notes.push(BirdNote { beat, beats: 1.0, midi: s.tuned(m), interval: sixth });
                beat += 2.0;
            }
            beat = (beat * 4.0).round() / 4.0 + if rng.next() < 0.3 { 1.0 } else { 0.0 };
        }
        parts.push(Part::Bird(BirdPart {
            name: format!("bird-{}", v + 1),
            pan: if n == 1 { 0.0 } else { -0.7 + 1.4 * v as f64 / (n - 1) as f64 },
            gain: 0.6,
            notes,
        }));
    }
    Ok(parts)
}

/// the note `steps` scale degrees from `m` (m snapped into the key first)
fn scale_step(s: &Settings, m: i32, steps: i32) -> i32 {
    // find m's degree (or the nearest below)
    let mut d = (m - s.key.tonic).div_euclid(12) * 7;
    while s.key.degree_semis(d + 1) <= m {
        d += 1;
    }
    while s.key.degree_semis(d) > m {
        d -= 1;
    }
    s.key.degree_semis(d + steps)
}

/// a voice's samples: pre-roll, the loop, then the tail
fn render_human(p: &HumanPart, bpm: f64, loop_beats: f64) -> Vec<f32> {
    let beat = 60.0 / bpm;
    let body = p.body.body();
    let mut v = HumanVoice::new(FS, body, 2);
    let first = p.events.iter().find_map(|e| e.midi).unwrap_or(57.0);
    let mut src = p.source.clone();
    src.f0 = Some(midi_to_hz(first));
    src.voiced = Some(true);
    v.set_source(&src);
    let mut t = PRE_ROLL + p.offset_beats * beat;
    let mut phrase: Vec<SungNote> = Vec::new();
    let mut start = t;
    let flush = |phrase: &mut Vec<SungNote>, start: f64, v: &mut HumanVoice| {
        if phrase.is_empty() {
            return;
        }
        let plan = plan_song(phrase, &SongOptions { body: p.body, bpm, t0: start, vibrato: p.vibrato });
        v.schedule(&plan.gestures);
        phrase.clear();
    };
    for e in &p.events {
        match (e.midi, &e.syl) {
            (Some(m), Some(syl)) => {
                if phrase.is_empty() {
                    start = t;
                }
                phrase.push(SungNote { midi: m, beats: e.beats, syl: syl.clone() });
            }
            _ => flush(&mut phrase, start, &mut v),
        }
        t += e.beats * beat;
    }
    flush(&mut phrase, start, &mut v);
    let n = ((PRE_ROLL + loop_beats * beat + TAIL) * FS).round() as usize;
    let mut out = vec![0f32; n];
    for block in out.chunks_mut(128) {
        v.render(block);
    }
    out
}

/// The pitch param that makes the songbird sound `hz`: the syrinx lands a few cents off its
/// target, so measure a held note and correct (twice).
fn bird_pitch(target_hz: f64, cache: &mut Vec<(f64, f64)>) -> f64 {
    if let Some(&(_, p)) = cache.iter().find(|(t, _)| (t / target_hz - 1.0).abs() < 1e-6) {
        return p;
    }
    let def = creature_by_id("songbird").expect("songbird");
    let ip = def.param_index("pitch").expect("pitch");
    let iv = def.param_index("voices").expect("voices");
    let mut p = target_hz;
    for _ in 0..2 {
        let mut v = CreatureVoice::new(def, FS, &[(ip, p), (iv, 0.0)]);
        v.set_gate(1.0);
        let mut buf = vec![0f32; (0.5 * FS) as usize];
        for b in buf.chunks_mut(128) {
            v.render(b);
        }
        let s = (0.25 * FS) as usize;
        let frame: Vec<f64> = buf[s..s + 4096].iter().map(|&x| x as f64).collect();
        let est = yin_pitch(&frame, FS, YinOpts { min_f0: Some(target_hz * 0.7), max_f0: Some(target_hz * 1.4), ..Default::default() });
        if est.f0 > 0.0 && est.aperiodicity < 0.5 {
            p *= target_hz / est.f0;
        }
    }
    cache.push((target_hz, p));
    p
}

fn render_bird(p: &BirdPart, bpm: f64, loop_beats: f64) -> Vec<f32> {
    let beat = 60.0 / bpm;
    let def = creature_by_id("songbird").expect("songbird");
    let ip = def.param_index("pitch").expect("pitch");
    let ii = def.param_index("interval").expect("interval");
    let iv = def.param_index("voices").expect("voices");
    let mut cache = Vec::new();
    // (time, pitch param, interval, gate)
    let mut changes: Vec<(f64, f64, f64, f64)> = Vec::new();
    for n in &p.notes {
        let t = PRE_ROLL + n.beat * beat;
        let hz = bird_pitch(midi_to_hz(n.midi), &mut cache);
        changes.push((t, hz, n.interval, 1.0));
        changes.push((t + n.beats * beat, hz, n.interval, 0.0));
    }
    changes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut v = CreatureVoice::new(def, FS, &[(iv, 0.0)]);
    v.set_gate(0.0);
    let n = ((PRE_ROLL + loop_beats * beat + TAIL) * FS).round() as usize;
    let mut out = vec![0f32; n];
    let mut pos = 0usize;
    for (t, hz, interval, gate) in changes {
        let at = ((t * FS).round() as usize).min(n);
        while pos < at {
            let end = (pos + 128).min(at);
            v.render(&mut out[pos..end]);
            pos = end;
        }
        if gate > 0.0 {
            // both voices when there's an interval, the left voice alone otherwise
            v.set_params(&[(ip, hz), (ii, interval), (iv, if interval != 0.0 { 0.5 } else { 0.0 })]);
        }
        v.set_gate(gate);
    }
    while pos < n {
        let end = (pos + 128).min(n);
        v.render(&mut out[pos..end]);
        pos = end;
    }
    out
}

/// Fold pre-roll and tail around the loop: sample i of the render lands at (i − pre-roll) mod L.
pub fn wrap(x: &[f32], loop_len: usize) -> Vec<f32> {
    let pre = (PRE_ROLL * FS).round() as isize;
    let mut out = vec![0f32; loop_len];
    for (i, &s) in x.iter().enumerate() {
        let j = (i as isize - pre).rem_euclid(loop_len as isize) as usize;
        out[j] += s;
    }
    out
}

fn pan_stereo(x: &[f32], pan: f64, gain: f64) -> Vec<f32> {
    let t = (pan.clamp(-1.0, 1.0) + 1.0) * std::f64::consts::FRAC_PI_4;
    let (l, r) = ((t.cos() * gain * std::f64::consts::SQRT_2) as f32, (t.sin() * gain * std::f64::consts::SQRT_2) as f32);
    let mut out = Vec::with_capacity(x.len() * 2);
    for &s in x {
        out.push(s * l);
        out.push(s * r);
    }
    out
}

fn flag<'a>(a: &'a [String], name: &str) -> Option<&'a str> {
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).map(String::as_str)
}

fn num_flag(a: &[String], name: &str, default: f64) -> Result<f64, String> {
    flag(a, name).map_or(Ok(default), |v| v.parse().map_err(|_| format!("{name}: not a number: {v:?}")))
}

fn words(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

pub fn run(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let kind = a.first().map(String::as_str).ok_or_else(|| format!("texture needs a kind\n\n{USAGE}"))?;
    let out_dir = a.get(1).filter(|x| !x.starts_with("--")).map_or("out/textures", String::as_str);
    let key = parse_key(flag(a, "--key").ok_or("--key is needed: \"A minor\", Am, F#, 8A …")?)?;
    let default_chords = if key.minor { "i VI III VII" } else { "I V vi IV" };
    let s = Settings {
        key,
        bpm: num_flag(a, "--bpm", 124.0)?,
        bars: num_flag(a, "--bars", 8.0)? as usize,
        tune_cents: num_flag(a, "--tune", 0.0)?,
        seed: num_flag(a, "--seed", 1.0)? as u32,
        chords: parse_chords(flag(a, "--chords").unwrap_or(default_chords))?,
        chord_bars: num_flag(a, "--chord-bars", 2.0)? as usize,
        voices: flag(a, "--voices").map(|v| v.parse().map_err(|_| format!("--voices: {v:?}"))).transpose()?,
        vowels: words(flag(a, "--vowels").unwrap_or("ah")),
        lyrics: flag(a, "--lyrics").map(words).unwrap_or_default(),
        breath: num_flag(a, "--breath", 0.35)?,
        vibrato: flag(a, "--vibrato").map(|v| v.parse().map_err(|_| format!("--vibrato: {v:?}"))).transpose()?,
        octave: num_flag(a, "--octave", 0.0)? as i32,
    };
    if s.bars == 0 {
        return Err("--bars must be at least 1".into());
    }
    let parts = match kind {
        "pad" => chordal(&s, false)?,
        "choir" => chordal(&s, true)?,
        "drone" => drone(&s)?,
        "chops" => chops(&s)?,
        "birds" => birds(&s)?,
        _ => return Err(format!("unknown texture {kind:?}\n\n{USAGE}")),
    };
    let t0 = std::time::Instant::now();
    let beats = s.beats();
    let loop_len = (beats * 60.0 / s.bpm * FS).round() as usize;
    let rendered: Vec<Vec<f32>> = std::thread::scope(|sc| {
        let handles: Vec<_> = parts
            .iter()
            .map(|p| {
                sc.spawn(move || match p {
                    Part::Human(h) => render_human(h, s.bpm, beats),
                    Part::Bird(b) => render_bird(b, s.bpm, beats),
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("render thread")).collect()
    });
    let stems: Vec<Vec<f32>> = parts
        .iter()
        .zip(&rendered)
        .map(|(p, x)| {
            let (pan, gain) = p.pan_gain();
            pan_stereo(&wrap(x, loop_len), pan, gain)
        })
        .collect();
    let mut mix = vec![0f32; loop_len * 2];
    for st in &stems {
        for (m, x) in mix.iter_mut().zip(st) {
            *m += x;
        }
    }
    let bad = mix.iter().filter(|x| !x.is_finite()).count();
    if bad > 0 {
        return Err(format!("{bad} non-finite samples in the mix (an airway went unstable)"));
    }
    // -1 dBFS peak; the stems take the same gain so they still sum to the loop
    let g = (0.891 / peak(&mix).max(1e-9)) as f32;
    // whole tempos as "124", others exactly ("125.52"), so the name never claims a grid it isn't on
    let bpm_name = if s.bpm.fract() == 0.0 { fixed(s.bpm, 0) } else { format!("{}", (s.bpm * 100.0).round() / 100.0) };
    let tag = format!("{kind}-{}-{bpm_name}bpm", s.key.short());
    let main = format!("{out_dir}/{tag}.wav");
    ctx.write_wav_stereo(&main, &mix.iter().map(|x| x * g).collect::<Vec<_>>())?;
    for (p, st) in parts.iter().zip(&stems) {
        ctx.write_wav_stereo(&format!("{out_dir}/{tag}-stems/{}.wav", p.name()), &st.iter().map(|x| x * g).collect::<Vec<_>>())?;
    }
    let info = serde_json::json!({
        "kind": kind,
        "key": format!("{} {}", NOTE_NAMES[s.key.tonic as usize], if s.key.minor { "minor" } else { "major" }),
        "camelot": s.key.camelot(),
        "tuneCents": s.tune_cents,
        "bpm": s.bpm,
        "bars": s.bars,
        "seed": s.seed,
        "chords": s.chord_spans().iter().map(|(c, b)| serde_json::json!({"chord": c.name, "beats": b})).collect::<Vec<_>>(),
        "parts": parts.iter().map(|p| match p {
            Part::Human(h) => serde_json::json!({
                "name": h.name, "body": {"female": h.body.female, "child": h.body.child}, "pan": h.pan,
                "notes": h.events.iter().map(|e| serde_json::json!([e.midi, e.beats, e.syl.as_ref().map(|s| s.text.clone())])).collect::<Vec<_>>(),
            }),
            Part::Bird(b) => serde_json::json!({
                "name": b.name, "pan": b.pan,
                "notes": b.notes.iter().map(|n| serde_json::json!([n.beat, n.midi, n.beats, n.interval])).collect::<Vec<_>>(),
            }),
        }).collect::<Vec<_>>(),
    });
    let info_path = format!("{out_dir}/{tag}.json");
    std::fs::write(&info_path, serde_json::to_string_pretty(&info).unwrap()).map_err(|e| format!("{info_path}: {e}"))?;
    println!(
        "{main}: {} bars at {bpm_name} bpm in {} ({}), {} voices, {} s loop, rendered in {} ms",
        s.bars,
        s.key.short(),
        s.key.camelot(),
        parts.len(),
        fixed(loop_len as f64 / FS, 2),
        t0.elapsed().as_millis()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_read_by_name_and_camelot() {
        assert_eq!(parse_key("A minor").unwrap(), Key { tonic: 9, minor: true });
        assert_eq!(parse_key("Am").unwrap(), Key { tonic: 9, minor: true });
        assert_eq!(parse_key("8A").unwrap(), Key { tonic: 9, minor: true });
        assert_eq!(parse_key("F#m").unwrap(), Key { tonic: 6, minor: true });
        assert_eq!(parse_key("Bbmaj").unwrap(), Key { tonic: 10, minor: false });
        assert_eq!(parse_key("C").unwrap(), Key { tonic: 0, minor: false });
        assert_eq!(parse_key("1B").unwrap(), Key { tonic: 11, minor: false });
        assert_eq!(parse_key("Cb").unwrap(), Key { tonic: 11, minor: false });
        assert!(parse_key("H minor").is_err());
        assert_eq!(parse_key("4A").unwrap().camelot(), "4A");
    }

    #[test]
    fn chords_take_their_quality_from_the_key() {
        let s = |chords: &str| Settings {
            key: parse_key("A minor").unwrap(),
            bpm: 120.0,
            bars: 8,
            tune_cents: 0.0,
            seed: 1,
            chords: parse_chords(chords).unwrap(),
            chord_bars: 2,
            voices: None,
            vowels: vec![],
            lyrics: vec![],
            breath: 0.3,
            vibrato: None,
            octave: 0,
        };
        let st = s("i VI III VII");
        let pcs = |c: &Chord| c.degrees.iter().map(|&d| st.key.degree_semis(d).rem_euclid(12)).collect::<Vec<_>>();
        assert_eq!(pcs(&st.chords[0]), vec![9, 0, 4]); // A C E
        assert_eq!(pcs(&st.chords[1]), vec![5, 9, 0]); // F A C
        assert_eq!(pcs(&st.chords[3]), vec![7, 11, 2]); // G B D
        assert_eq!(st.chord_spans().len(), 4);
        assert!(st.chord_spans().iter().all(|(_, b)| *b == 8.0));
        assert!(parse_chords("x").is_err());
    }

    #[test]
    fn every_voice_sings_only_chord_tones_and_the_bass_sings_roots() {
        let s = Settings {
            key: parse_key("F# minor").unwrap(),
            bpm: 124.0,
            bars: 8,
            tune_cents: 0.0,
            seed: 3,
            chords: parse_chords("i iv VI v").unwrap(),
            chord_bars: 2,
            voices: None,
            vowels: vec![],
            lyrics: vec![],
            breath: 0.3,
            vibrato: None,
            octave: 0,
        };
        let centers: Vec<f64> = (0..6).map(|v| 45.0 + 27.0 * v as f64 / 5.0).collect();
        let lines = voice_lines(&s, 6, &centers);
        for (k, (chord, _)) in s.chord_spans().iter().enumerate() {
            let pcs: Vec<i32> = chord.degrees.iter().map(|&d| s.key.degree_semis(d).rem_euclid(12)).collect();
            assert_eq!(lines[0][k].0.rem_euclid(12), pcs[0], "bass on the root");
            for line in &lines {
                assert!(pcs.contains(&line[k].0.rem_euclid(12)));
            }
        }
    }

    #[test]
    fn scale_steps_stay_in_the_key() {
        let s = Settings {
            key: parse_key("C major").unwrap(),
            bpm: 120.0,
            bars: 1,
            tune_cents: 0.0,
            seed: 1,
            chords: parse_chords("I").unwrap(),
            chord_bars: 1,
            voices: None,
            vowels: vec![],
            lyrics: vec![],
            breath: 0.3,
            vibrato: None,
            octave: 0,
        };
        assert_eq!(scale_step(&s, 60, 1), 62);
        assert_eq!(scale_step(&s, 64, 1), 65);
        assert_eq!(scale_step(&s, 71, 1), 72);
        assert_eq!(scale_step(&s, 60, -1), 59);
        assert_eq!(scale_step(&s, 60, 2), 64);
    }

    #[test]
    fn wrapping_keeps_every_sample_and_lands_the_pre_roll_at_the_end() {
        let pre = (PRE_ROLL * FS).round() as usize;
        let l = 1000;
        let mut x = vec![0f32; pre + l + 500];
        x[pre - 1] = 1.0; // just before the downbeat
        x[pre] = 2.0; // the downbeat
        x[pre + l + 3] = 3.0; // tail
        let w = wrap(&x, l);
        assert_eq!(w[l - 1], 1.0);
        assert_eq!(w[0], 2.0);
        assert_eq!(w[3], 3.0);
        assert_eq!(w.iter().sum::<f32>(), 6.0);
    }
}
