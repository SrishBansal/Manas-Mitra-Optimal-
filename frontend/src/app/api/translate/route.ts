import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

type SupportedLang = 'en' | 'hi' | 'mr' | 'kn';

interface TranslateRequestBody {
  text?: string;
  texts?: string[];
  targetLang: SupportedLang;
  sourceLang?: SupportedLang; // required for providers that don't support auto
}

const LIBRE_ENDPOINTS = [
  'https://libretranslate.com/translate',
  'https://translate.mentality.rip/translate',
  'https://libretranslate.de/translate'
];

function toFormBody(params: Record<string, string>) {
  return Object.entries(params)
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
    .join('&');
}

async function translateWithLibre(text: string, sourceLang: SupportedLang, targetLang: SupportedLang, signal?: AbortSignal): Promise<string | null> {
  const body = toFormBody({ q: text, source: sourceLang, target: targetLang, format: 'text', api_key: '' });
  for (const endpoint of LIBRE_ENDPOINTS) {
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body,
        signal
      });
      if (!res.ok) continue;
      const data = (await res.json()) as { translatedText?: string };
      if (data && typeof data.translatedText === 'string') return data.translatedText;
    } catch {
      // try next endpoint
    }
  }
  return null;
}

async function translateWithMyMemory(text: string, sourceLang: SupportedLang, targetLang: SupportedLang, signal?: AbortSignal): Promise<string | null> {
  // MyMemory expects langpair like en|hi
  const url = `https://api.mymemory.translated.net/get?q=${encodeURIComponent(text)}&langpair=${encodeURIComponent(sourceLang)}|${encodeURIComponent(targetLang)}`;
  try {
    const res = await fetch(url, { method: 'GET', signal, headers: { 'Cache-Control': 'no-store' } });
    if (!res.ok) return null;
    const data = (await res.json()) as { responseData?: { translatedText?: string } };
    const t = data?.responseData?.translatedText;
    return typeof t === 'string' ? t : null;
  } catch {
    return null;
  }
}

async function translateBatch(texts: string[], sourceLang: SupportedLang, targetLang: SupportedLang): Promise<string[]> {
  if (sourceLang === targetLang) return texts;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12000);
  try {
    const results = await Promise.all(
      texts.map(async (t) => {
        // Try LibreTranslate first
        const libre = await translateWithLibre(t, sourceLang, targetLang, controller.signal);
        if (libre) return libre;
        // Fallback to MyMemory
        const mem = await translateWithMyMemory(t, sourceLang, targetLang, controller.signal);
        return mem ?? t;
      })
    );
    return results;
  } finally {
    clearTimeout(timeout);
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as TranslateRequestBody;
    const { text, texts, targetLang, sourceLang } = body;

    if (!targetLang || !['en', 'hi', 'mr', 'kn'].includes(targetLang)) {
      return NextResponse.json(
        { error: 'Invalid or missing targetLang. Use one of en, hi, mr, kn.' },
        { status: 400 }
      );
    }

    const effectiveSource = (sourceLang as SupportedLang) || 'en';
    if (!['en', 'hi', 'mr', 'kn'].includes(effectiveSource)) {
      return NextResponse.json(
        { error: 'Invalid sourceLang. Use one of en, hi, mr, kn.' },
        { status: 400 }
      );
    }

    const inputArray: string[] = Array.isArray(texts)
      ? texts.filter((t) => typeof t === 'string')
      : typeof text === 'string'
      ? [text]
      : [];

    if (inputArray.length === 0) {
      return NextResponse.json({ translations: [] });
    }

    const translations = await translateBatch(inputArray, effectiveSource, targetLang);

    return NextResponse.json({ translations });
  } catch (error) {
    return NextResponse.json(
      { error: 'Translation failed' },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({ message: 'Translate API is running' });
}


