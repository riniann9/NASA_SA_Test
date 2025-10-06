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
  // Extract key features for image generation
  const orbitalPeriod = planetData.orbital_period || "unknown"
  const planetaryRadius = planetData.planetary_radius || "unknown"
  const equilibriumTemp = planetData.equilibrium_temperature || "unknown"
  const stellarTemp = planetData.stellar_effective_temperature || "unknown"
  const stellarRadius = planetData.stellar_radius || "unknown"
  const insolationFlux = planetData.insolation_flux || "unknown"
  const transitDepth = planetData.transit_depth || "unknown"
  const transitDuration = planetData.transit_duration || "unknown"
  const impactParameter = planetData.impact_parameter || "unknown"
  const transitSNR = planetData.transit_snr || "unknown"

  // Determine planet characteristics based on data
  const isHabitable = parseFloat(equilibriumTemp) > 200 && parseFloat(equilibriumTemp) < 350
  const isSuperEarth = parseFloat(planetaryRadius) > 1.0 && parseFloat(planetaryRadius) < 2.0
  const isHotPlanet = parseFloat(equilibriumTemp) > 500
  const isColdPlanet = parseFloat(equilibriumTemp) < 200
  
  // Determine stellar characteristics
  const isKStar = parseFloat(stellarTemp) > 3500 && parseFloat(stellarTemp) < 5000
  const isGStar = parseFloat(stellarTemp) > 5000 && parseFloat(stellarTemp) < 6000
  const isMStar = parseFloat(stellarTemp) < 3500

  return `Create a detailed, scientifically accurate visualization of an exoplanet system based on these Kepler dataset parameters:

**Planetary Properties:**
- Orbital Period: ${orbitalPeriod} days
- Planetary Radius: ${planetaryRadius} Earth radii (${isSuperEarth ? 'Super-Earth' : 'Earth-sized'})
- Equilibrium Temperature: ${equilibriumTemp} K (${isHotPlanet ? 'Hot' : isColdPlanet ? 'Cold' : isHabitable ? 'Potentially Habitable' : 'Temperate'})
- Insolation Flux: ${insolationFlux} Earth flux

**Stellar Properties:**
- Stellar Effective Temperature: ${stellarTemp} K (${isKStar ? 'K-type star' : isGStar ? 'G-type star' : isMStar ? 'M-type star' : 'Unknown type'})
- Stellar Radius: ${stellarRadius} Solar radii

**Transit Properties:**
- Transit Depth: ${transitDepth} ppm
- Transit Duration: ${transitDuration} hours
- Impact Parameter: ${impactParameter}
- Signal-to-Noise Ratio: ${transitSNR}

**Visualization Requirements:**
1. Show the exoplanet as the main subject with accurate size relative to Earth (${planetaryRadius}x Earth size)
2. Include the host star in the background with appropriate color based on stellar temperature (${stellarTemp}K = ${isKStar ? 'orange' : isGStar ? 'yellow-white' : isMStar ? 'red' : 'white'} color)
3. Show the orbital relationship and distance between planet and star (${orbitalPeriod}-day orbit)
4. Include realistic atmospheric features based on temperature (${equilibriumTemp}K surface temperature)
5. Add space environment with stars and cosmic background
6. Use scientifically accurate colors and lighting
7. Show the planet's surface features that would be consistent with the given parameters
8. Include realistic atmospheric composition based on temperature and insolation
9. Make it photorealistic and space-like with proper lighting from the host star
10. The image should be suitable for scientific presentation and clearly show the planet's characteristics

Generate a high-quality, detailed image that represents this exoplanet system accurately based on the actual Kepler data parameters.`
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
