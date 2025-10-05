import { NextRequest, NextResponse } from "next/server"
import { GoogleGenerativeAI } from "@google/generative-ai"

export async function POST(req: NextRequest) {
  try {
    const { prompt } = await req.json()
    if (!prompt || typeof prompt !== "string") {
      return NextResponse.json({ error: "Prompt is required" }, { status: 400 })
    }

    const apiKey = process.env.GEMINI_API_KEY
    if (!apiKey) {
      return NextResponse.json({ error: "Missing GEMINI_API_KEY" }, { status: 500 })
    }

    const genAI = new GoogleGenerativeAI(apiKey)
    // Imagen 3 model name
    const model = genAI.getGenerativeModel({ model: "imagen-3.0-generate-001" })

    const res = await model.generateContent({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      // Ask for image output
      generationConfig: { responseMimeType: "image/png" },
    } as any)

    // SDK returns a binary image in the response; extract inline data if present
    const candidates: any[] = (res as any).response?.candidates || []
    const parts: any[] = candidates[0]?.content?.parts || []
    const imagePart = parts.find((p: any) => p.inlineData?.data)
    if (!imagePart) {
      return NextResponse.json({ error: "No image returned" }, { status: 502 })
    }
    const imageBase64 = imagePart.inlineData.data as string
    const mime = imagePart.inlineData.mimeType || "image/png"
    const dataUrl = `data:${mime};base64,${imageBase64}`
    return NextResponse.json({ imageUrl: dataUrl })
  } catch (err) {
    console.error("/api/generate-planet-image error", err)
    return NextResponse.json({ error: "Failed to generate image" }, { status: 500 })
  }
}


