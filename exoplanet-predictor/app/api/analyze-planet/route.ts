import { NextRequest, NextResponse } from "next/server"
import { GoogleGenerativeAI } from "@google/generative-ai"

export async function POST(req: NextRequest) {
  try {
    const { formData } = await req.json()

    if (!formData || typeof formData !== "object") {
      return NextResponse.json({ error: "Invalid payload" }, { status: 400 })
    }

    const apiKey = process.env.GEMINI_API_KEY
    if (!apiKey) {
      return NextResponse.json({ error: "Missing GEMINI_API_KEY" }, { status: 500 })
    }

    const genAI = new GoogleGenerativeAI(apiKey)
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash-001" })

    const prompt = buildPrompt(formData)

    const result = await model.generateContent({ contents: [{ role: "user", parts: [{ text: prompt }] }] })
    const text = result.response.text()

    return NextResponse.json({ analysis: text })
  } catch (err) {
    console.error("/api/analyze-planet error", err)
    return NextResponse.json({ error: "Failed to analyze planet data" }, { status: 500 })
  }
}

function buildPrompt(formData: Record<string, unknown>): string {
  const entries = Object.entries(formData)
    .map(([key, value]) => `${key}: ${String(value ?? "")}`)
    .join("\n")

  return `You are an astrophysicist AI analyzing potential exoplanet observations.
Given the following features, provide:
1) A clear classification if this is likely an exoplanet (Yes/No/Likely/Unlikely).
2) A confidence percentage (0-100%).
3) A detailed scientific explanation (4-8 bullet-style paragraphs) referencing specific features.
4) The likely planet type (e.g., terrestrial, super-Earth, gas giant, ice giant) and habitability discussion.

Features:\n${entries}

Output in well-structured plain text suitable for UI display.`
}

