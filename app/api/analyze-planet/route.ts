import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { planetData } = body

    // Format the data for Gemini analysis
    const formattedData = formatPlanetDataForGemini(planetData)
    
    // Create the prompt for Gemini
    const prompt = createGeminiPrompt(formattedData)
    
    // Call Gemini API
    const geminiResponse = await callGeminiAPI(prompt)
    
    // Parse and format the response
    const analysisResult = parseGeminiResponse(geminiResponse)
    
    return NextResponse.json(analysisResult)
  } catch (error) {
    console.error('Error analyzing planet:', error)
    return NextResponse.json(
      { error: 'Failed to analyze planet data' },
      { status: 500 }
    )
  }
}

function formatPlanetDataForGemini(planetData: any) {
  // Format Kepler dataset fields for Gemini analysis
  return {
    "Orbital Period [days]": planetData.orbital_period || "Unknown",
    "Transit Epoch [BKJD]": planetData.transit_epoch || "Unknown",
    "Impact Parameter": planetData.impact_parameter || "Unknown",
    "Transit Duration [hrs]": planetData.transit_duration || "Unknown",
    "Transit Depth [ppm]": planetData.transit_depth || "Unknown",
    "Planetary Radius [Earth radii]": planetData.planetary_radius || "Unknown",
    "Equilibrium Temperature [K]": planetData.equilibrium_temperature || "Unknown",
    "Insolation Flux [Earth flux]": planetData.insolation_flux || "Unknown",
    "Transit Signal-to-Noise": planetData.transit_snr || "Unknown",
    "TCE Planet Number": planetData.tce_planet_number || "Unknown",
    "Stellar Effective Temperature [K]": planetData.stellar_effective_temperature || "Unknown",
    "Stellar Surface Gravity [log10(cm/s**2)]": planetData.stellar_surface_gravity || "Unknown",
    "Stellar Radius [Solar radii]": planetData.stellar_radius || "Unknown",
    "RA [decimal degrees]": planetData.ra || "Unknown",
    "Dec [decimal degrees]": planetData.dec || "Unknown",
    "Kepler-band [mag]": planetData.kepler_band_mag || "Unknown"
  }
}

function calculateTransitDuration(planetData: any): string {
  if (!planetData.orbital_period || !planetData.stellar_radius) return "Unknown"
  
  const period = parseFloat(planetData.orbital_period)
  const stellarRadius = parseFloat(planetData.stellar_radius)
  
  // Simplified transit duration calculation
  const duration = (period * stellarRadius * 24) / (Math.PI * 2)
  return duration.toFixed(2)
}

function calculateTransitDepth(planetData: any): string {
  if (!planetData.planet_radius || !planetData.stellar_radius) return "Unknown"
  
  const planetRadius = parseFloat(planetData.planet_radius)
  const stellarRadius = parseFloat(planetData.stellar_radius)
  
  // Transit depth in ppm: (Rp/Rs)^2 * 1e6
  const depth = Math.pow(planetRadius / stellarRadius, 2) * 1e6
  return depth.toFixed(2)
}

function calculateSNR(planetData: any): string {
  // Simplified SNR calculation based on transit depth and stellar properties
  if (!planetData.planet_radius || !planetData.stellar_radius) return "Unknown"
  
  const planetRadius = parseFloat(planetData.planet_radius)
  const stellarRadius = parseFloat(planetData.stellar_radius)
  const depth = Math.pow(planetRadius / stellarRadius, 2)
  
  // Rough SNR estimate
  const snr = Math.sqrt(depth * 1000) // Simplified calculation
  return snr.toFixed(2)
}

function calculateStellarSurfaceGravity(planetData: any): string {
  if (!planetData.stellar_mass || !planetData.stellar_radius) return "Unknown"
  
  const mass = parseFloat(planetData.stellar_mass)
  const radius = parseFloat(planetData.stellar_radius)
  
  // log10(g) = log10(G*M/R^2) in cgs units
  const g = 4.44 + Math.log10(mass) - 2 * Math.log10(radius)
  return g.toFixed(2)
}

function createGeminiPrompt(formattedData: any): string {
  const dataString = Object.entries(formattedData)
    .map(([key, value]) => `${key}: ${value}`)
    .join(', ')

  return `You are an expert exoplanet scientist. Analyze this exoplanet data and determine if it represents a confirmed exoplanet. Respond with ONLY the JSON format below, no additional text.

Data: ${dataString}

Required JSON format (respond with exactly this structure):
{
  "answer": true,
  "most_important_features": [
    {
      "feature1": "Feature Name (with specific values): Detailed explanation of why this feature is important for exoplanet classification.",
      "Relevance": "Relevance level - High/Medium/Low with brief explanation of why this feature matters for exoplanet identification."
    },
    {
      "feature2": "Feature Name (with specific values): Detailed explanation of why this feature is important for exoplanet classification.",
      "Relevance": "Relevance level - High/Medium/Low with brief explanation of why this feature matters for exoplanet identification."
    },
    {
      "feature3": "Feature Name (with specific values): Detailed explanation of why this feature is important for exoplanet classification.",
      "Relevance": "Relevance level - High/Medium/Low with brief explanation of why this feature matters for exoplanet identification."
    },
    {
      "feature4": "Feature Name (with specific values): Detailed explanation of why this feature is important for exoplanet classification.",
      "Relevance": "Relevance level - High/Medium/Low with brief explanation of why this feature matters for exoplanet identification."
    },
    {
      "feature5": "Feature Name (with specific values): Detailed explanation of why this feature is important for exoplanet classification.",
      "Relevance": "Relevance level - High/Medium/Low with brief explanation of why this feature matters for exoplanet identification."
    }
  ]
}`
}

async function callGeminiAPI(prompt: string): Promise<any> {
  const apiKey = process.env.GEMINI_API_KEY
  
  console.log('GEMINI_API_KEY exists:', !!apiKey)
  console.log('GEMINI_API_KEY length:', apiKey?.length || 0)
  
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY environment variable is not set')
  }

  const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=${apiKey}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      contents: [{
        parts: [{
          text: prompt
        }]
      }],
      generationConfig: {
        temperature: 0.1,
        topK: 1,
        topP: 0.8,
        maxOutputTokens: 4096,
      }
    })
  })

  if (!response.ok) {
    throw new Error(`Gemini API error: ${response.status} ${response.statusText}`)
  }

  const data = await response.json()
  return data
}

function parseGeminiResponse(geminiResponse: any): any {
  try {
    console.log('Gemini response received')
    
    // Extract the text content from Gemini response
    const textContent = geminiResponse.candidates?.[0]?.content?.parts?.[0]?.text
    
    if (!textContent) {
      console.log('No text content found. Full response structure:', JSON.stringify(geminiResponse, null, 2))
      throw new Error('No text content in Gemini response')
    }

    console.log('Text content from Gemini:', textContent)

    // Try to extract JSON from the response (handle both raw JSON and markdown code blocks)
    let jsonMatch = textContent.match(/\{[\s\S]*\}/)
    if (!jsonMatch) {
      // Try to extract from markdown code blocks
      const codeBlockMatch = textContent.match(/```(?:json)?\s*(\{[\s\S]*?\})\s*```/)
      if (codeBlockMatch) {
        jsonMatch = [codeBlockMatch[1]]
      }
    }

    if (!jsonMatch) {
      console.log('No JSON found in response. Full text:', textContent)
      throw new Error('No JSON found in Gemini response')
    }

    const analysis = JSON.parse(jsonMatch[0])
    console.log('Parsed analysis:', analysis)
    
    // Transform to our expected format
    return {
      isExoplanet: analysis.answer || true, // Default to true if not specified
      confidence: analysis.answer ? 0.85 : 0.15, // Mock confidence based on answer
      explanation: `Based on the analysis of ${analysis.most_important_features?.length || 5} key features, this object ${analysis.answer ? 'is classified as an exoplanet' : 'is not classified as an exoplanet'}.`,
      topFeatures: (analysis.most_important_features || []).map((feature: any, index: number) => ({
        name: Object.keys(feature).find(key => key !== 'Relevance') || `Feature ${index + 1}`,
        impact: 0.8 - (index * 0.1), // Decreasing impact
        value: "Analyzed",
        reasoning: Object.values(feature).find(value => typeof value === 'string' && !value.includes('High') && !value.includes('Medium') && !value.includes('Low')) as string || Object.values(feature)[0] as string
      })),
      geminiAnalysis: analysis
    }
  } catch (error) {
    console.error('Error parsing Gemini response:', error)
    throw new Error('Failed to parse Gemini response')
  }
}
