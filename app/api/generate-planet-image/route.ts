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

  return `NASA SPACE APP CHALLENGE - EXOPLANET VISUALIZATION PROMPT

Create a scientifically accurate, photorealistic visualization of an exoplanet system discovered by NASA's Kepler Space Telescope. This visualization should represent the cutting-edge discoveries in exoplanet science and be suitable for NASA's Space App Challenge presentation.

**KEPLER DATASET PARAMETERS:**
- Orbital Period: ${orbitalPeriod} days (${orbitalPeriod > 365 ? 'Outer system' : orbitalPeriod < 10 ? 'Inner system' : 'Mid-range orbit'})
- Planetary Radius: ${planetaryRadius} Earth radii (${isSuperEarth ? 'Super-Earth category' : 'Earth-sized'})
- Equilibrium Temperature: ${equilibriumTemp} K (${isHotPlanet ? 'Hot Jupiter-like' : isColdPlanet ? 'Ice Giant-like' : isHabitable ? 'Potentially Habitable Zone' : 'Temperate Zone'})
- Insolation Flux: ${insolationFlux} Earth flux (${parseFloat(insolationFlux) > 1 ? 'High insolation' : 'Low insolation'})

**STELLAR SYSTEM CHARACTERISTICS:**
- Host Star Type: ${isKStar ? 'K-type (Orange Dwarf)' : isGStar ? 'G-type (Solar-like)' : isMStar ? 'M-type (Red Dwarf)' : 'Unknown spectral type'}
- Stellar Temperature: ${stellarTemp} K
- Stellar Radius: ${stellarRadius} Solar radii
- Transit Depth: ${transitDepth} ppm (${parseFloat(transitDepth) > 1000 ? 'Deep transit' : 'Shallow transit'})
- Transit Duration: ${transitDuration} hours
- Signal-to-Noise Ratio: ${transitSNR} (${parseFloat(transitSNR) > 10 ? 'High confidence detection' : 'Moderate confidence'})

**SCIENTIFIC VISUALIZATION REQUIREMENTS:**

1. **EXOPLANET PORTRAIT:**
   - Show the exoplanet as the primary subject with accurate scale (${planetaryRadius}x Earth size)
   - Surface features consistent with ${equilibriumTemp}K temperature
   - Atmospheric composition based on stellar insolation and temperature
   - Realistic cloud patterns and atmospheric layers
   - ${isHabitable ? 'Potential for liquid water and life-supporting conditions' : 'Extreme environment characteristics'}

2. **STELLAR ENVIRONMENT:**
   - Host star with scientifically accurate color (${stellarTemp}K = ${isKStar ? 'orange-red' : isGStar ? 'yellow-white' : isMStar ? 'deep red' : 'white'})
   - Proper stellar luminosity and size relative to the planet
   - Stellar corona and solar wind effects
   - Habitable zone boundaries if applicable

3. **ORBITAL DYNAMICS:**
   - Show the ${orbitalPeriod}-day orbital relationship
   - Transit geometry with impact parameter ${impactParameter}
   - Orbital plane and inclination
   - Tidal effects and gravitational interactions

4. **SPACE ENVIRONMENT:**
   - Deep space background with distant stars
   - Cosmic dust and nebula elements
   - Proper lighting and shadows from the host star
   - Space-time curvature effects (subtle gravitational lensing)

5. **SCIENTIFIC ACCURACY:**
   - Photorealistic rendering suitable for NASA presentation
   - Accurate color temperature and lighting
   - Realistic atmospheric scattering and absorption
   - Proper scale relationships between all objects
   - Scientific instrumentation context (Kepler telescope silhouette)

6. **NASA SPACE APP CHALLENGE CONTEXT:**
   - Professional scientific visualization quality
   - Educational value for public outreach
   - Cutting-edge exoplanet discovery representation
   - Integration with NASA's exoplanet database
   - Suitable for mission planning and scientific communication

**TECHNICAL SPECIFICATIONS:**
- High-resolution, photorealistic rendering
- Scientific accuracy in all physical parameters
- Professional presentation quality
- Suitable for NASA Space App Challenge submission
- Educational and inspirational for space science outreach

Generate a stunning, scientifically accurate visualization that showcases this exoplanet discovery as a testament to NASA's Kepler mission and the future of exoplanet exploration.`
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
