import { GoogleGenerativeAI } from '@google/generative-ai'
import { NextResponse } from 'next/server'

const MODEL_NAME = 'gemini-2.5-pro'

export async function POST(req: Request) {
  try {
    const { planetData } = await req.json()

    const prompt = createImagePrompt(planetData)
    const imageUrl = await generatePlanetImage(prompt)

    return NextResponse.json({ imageUrl })
  } catch (error: any) {
    console.error('Error in generate-planet-image API:', error)
    return new NextResponse(
      JSON.stringify({ error: error.message || 'Failed to generate planet image' }),
      { status: 500 }
    )
  }
}

function createImagePrompt(planetData: any): string {
  // Extract key features for image generation using NASA repository approach
  const temp = Number.parseFloat(planetData.equilibrium_temperature) || 250
  const radius = Number.parseFloat(planetData.planetary_radius) || 1
  const hasRings = planetData.ring_system === true || planetData.ring_system === "true"

  let description = "exoplanet in space"

  // Temperature-based appearance (matching NASA repository logic)
  if (temp < 200) {
    description += " frozen ice planet blue white"
  } else if (temp < 300) {
    description += " earth-like planet blue green"
  } else if (temp < 500) {
    description += " hot desert planet orange red"
  } else {
    description += " lava planet glowing red orange"
  }

  // Size-based (matching NASA repository logic)
  if (radius > 3) {
    description += " gas giant"
  } else if (radius > 1.5) {
    description += " super earth"
  }

  // Rings (matching NASA repository logic)
  if (hasRings) {
    description += " with ring system"
  }

  description += " realistic space background stars"

  return description
}

async function generatePlanetImage(prompt: string): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY

  if (!apiKey) {
    throw new Error('GEMINI_API_KEY environment variable is not set')
  }

  const genAI = new GoogleGenerativeAI(apiKey)
  const model = genAI.getGenerativeModel({ model: MODEL_NAME })

  try {
    // Use Gemini to generate a detailed image description
    const imageDescriptionResult = await model.generateContent({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: {
        temperature: 0.8,
        topK: 1,
        topP: 0.9,
        maxOutputTokens: 2048,
      },
    })

    const imageDescription = imageDescriptionResult.response.text()
    console.log('Generated image description:', imageDescription)

    // For now, return a placeholder that represents the generated description
    // In a full implementation, you would use Gemini's actual image generation
    const placeholderImageUrl = `https://images.unsplash.com/photo-1446776877081-d282a0f896e2?w=800&h=600&fit=crop&crop=center&auto=format&q=80&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D`
    
    return placeholderImageUrl
  } catch (error) {
    console.error('Error generating planet image:', error)
    // Return a fallback image
    return `https://images.unsplash.com/photo-1446776877081-d282a0f896e2?w=800&h=600&fit=crop&crop=center&auto=format&q=80&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D`
  }
}
