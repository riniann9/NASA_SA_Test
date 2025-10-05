"use client"

import { useEffect, useMemo, useState, Suspense } from "react"
import { useSearchParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, CheckCircle2, XCircle, TrendingUp, Home } from "lucide-react"
import Link from "next/link"
import { Progress } from "@/components/ui/progress"

type AnalysisResult = {
  isExoplanet: boolean
  confidence: number
  explanation: string
  aiAnalysis?: string
  topFeatures: Array<{
    name: string
    impact: number
    value: string | number
    reasoning: string
  }>
  planetData: any
  source: string
}

function ResultsContent() {
  const searchParams = useSearchParams()
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)

  // Compute the planet image prompt only when result is available
  const planetImageQuery = useMemo(() => {
    if (!result) return null
    const normalized = normalizePlanetData(result.planetData)
    return generatePlanetImageQuery(normalized)
  }, [result])

  // Always register this hook; guard inside on missing data
  useEffect(() => {
    let isMounted = true
    async function loadImage() {
      if (!planetImageQuery) return
      try {
        const res = await fetch('/api/generate-planet-image', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: planetImageQuery })
        })
        if (!res.ok) throw new Error('image api failed')
        const data = await res.json()
        if (isMounted && data.imageUrl) setImageUrl(data.imageUrl)
      } catch {
        // fall back to public AI image generator
        if (isMounted) setImageUrl(`https://image.pollinations.ai/prompt/${encodeURIComponent(planetImageQuery)}?width=400&height=400`)
      }
    }
    loadImage()
    return () => { isMounted = false }
  }, [planetImageQuery])

  useEffect(() => {
    const planetDataStr = searchParams.get("planetData")
    const aiAnalysisStr = searchParams.get("aiAnalysis")
    const source = searchParams.get("source")
    const error = searchParams.get("error")

    if (planetDataStr) {
      const planetData = JSON.parse(planetDataStr)
      const aiAnalysis = aiAnalysisStr ? JSON.parse(aiAnalysisStr) : null

      // Use AI analysis if available, otherwise fallback to mock analysis
      const analysis = aiAnalysis 
        ? analyzePlanetDataWithAI(planetData, aiAnalysis, source || "unknown")
        : analyzePlanetData(planetData, source || "unknown")
      
      setResult(analysis)
    }
  }, [searchParams])

  if (!result) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <p className="text-muted-foreground">Loading results...</p>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Cosmic background effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div
          className={`absolute top-1/4 right-1/4 w-96 h-96 ${result.isExoplanet ? "bg-primary/10" : "bg-destructive/10"} rounded-full blur-[120px] animate-pulse`}
        />
        <div
          className={`absolute bottom-1/4 left-1/4 w-96 h-96 ${result.isExoplanet ? "bg-accent/10" : "bg-muted/10"} rounded-full blur-[120px] animate-pulse`}
          style={{ animationDelay: "1s" }}
        />
      </div>

      <div className="relative z-10 container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <Link href="/">
            <Button variant="outline" size="sm" className="gap-2 bg-transparent">
              <Home className="w-4 h-4" />
              Back to Home
            </Button>
          </Link>

          <Badge variant="outline" className="text-sm">
            Source: {result.source === "existing" ? "Existing Data" : "New Prediction"}
          </Badge>
        </div>

        {/* Main Result Card */}
        <Card
          className={`p-8 mb-8 border-4 ${result.isExoplanet ? "border-primary bg-primary/5" : "border-destructive bg-destructive/5"}`}
        >
          <div className="flex flex-col md:flex-row items-center gap-8">
            {/* Result Icon */}
            <div className="flex-shrink-0">
              {result.isExoplanet ? (
                <CheckCircle2 className="w-32 h-32 text-primary animate-pulse" />
              ) : (
                <XCircle className="w-32 h-32 text-destructive animate-pulse" />
              )}
            </div>

            {/* Result Text */}
            <div className="flex-1 text-center md:text-left">
              <h1 className="text-5xl font-bold mb-4 text-balance">
                {result.isExoplanet ? (
                  <span className="text-primary">Exoplanet Detected!</span>
                ) : (
                  <span className="text-destructive">Not an Exoplanet</span>
                )}
              </h1>
              <p className="text-xl text-muted-foreground mb-4">
                Confidence Level: <span className="font-bold text-foreground">{result.confidence}%</span>
              </p>
              <Progress value={result.confidence} className="h-3" />
            </div>
          </div>
        </Card>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* AI Explanation */}
          <Card className="p-6 bg-card/50 backdrop-blur border-2 border-border">
            <h2 className="text-2xl font-bold mb-4 text-primary flex items-center gap-2">
              <TrendingUp className="w-6 h-6" />
              {result.aiAnalysis ? 'Gemini AI Analysis' : 'AI Analysis'}
            </h2>
            {result.aiAnalysis ? (
              <div className="space-y-4">
                <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
                  <h3 className="font-semibold text-primary mb-2">Detailed Scientific Analysis:</h3>
                  <div className="text-foreground leading-relaxed whitespace-pre-line">
                    {result.aiAnalysis}
                  </div>
                </div>
                <div className="p-4 bg-muted/50 border border-border rounded-lg">
                  <h3 className="font-semibold text-foreground mb-2">Summary:</h3>
                  <p className="text-foreground leading-relaxed">{result.explanation}</p>
                </div>
              </div>
            ) : (
              <p className="text-foreground leading-relaxed">{result.explanation}</p>
            )}
          </Card>

          {/* Planet Visualization */}
          <Card className="p-6 bg-card/50 backdrop-blur border-2 border-border">
            <h2 className="text-2xl font-bold mb-4 text-primary">Planet Visualization</h2>
            <div className="aspect-square rounded-lg overflow-hidden bg-background/50 border border-border">
              {imageUrl ? (
                <img
                  src={imageUrl}
                  alt="AI Generated Planet"
                  className="w-full h-full object-cover"
                  referrerPolicy="no-referrer"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-muted-foreground text-sm">Generating image…</div>
              )}
            </div>
            <p className="text-sm text-muted-foreground mt-3 text-center">
              AI-generated visualization based on features
            </p>
          </Card>
        </div>

        {/* Feature Impact Analysis */}
        <Card className="p-6 mt-8 bg-card/50 backdrop-blur border-2 border-border">
          <h2 className="text-2xl font-bold mb-6 text-primary">Key Features Impact</h2>
          <div className="space-y-6">
            {result.topFeatures.map((feature, index) => (
              <div key={index} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Badge variant="outline" className="text-lg px-3 py-1">
                      #{index + 1}
                    </Badge>
                    <span className="font-semibold text-foreground">{feature.name}</span>
                  </div>
                  <span className="text-sm text-muted-foreground">
                    Value: <span className="font-bold text-foreground">{feature.value}</span>
                  </span>
                </div>

                <div className="flex items-center gap-3">
                  <Progress value={feature.impact} className="flex-1 h-3" />
                  <span className="text-sm font-bold text-foreground w-12">{feature.impact}%</span>
                </div>

                <p className="text-sm text-muted-foreground pl-14">{feature.reasoning}</p>
              </div>
            ))}
          </div>
        </Card>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center mt-8">
          <Link href="/existing">
            <Button size="lg" variant="outline" className="gap-2 min-w-[200px] bg-transparent">
              <ArrowLeft className="w-5 h-5" />
              Explore Existing Planets
            </Button>
          </Link>
          <Link href="/new">
            <Button size="lg" className="gap-2 min-w-[200px] bg-primary hover:bg-primary/90">
              Create New Prediction
            </Button>
          </Link>
        </div>
      </div>
    </div>
  )
}

export default function ResultsPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-background flex items-center justify-center">
          <p className="text-muted-foreground">Loading results...</p>
        </div>
      }
    >
      <ResultsContent />
    </Suspense>
  )
}

// AI analysis function that processes Gemini response
function analyzePlanetDataWithAI(planetData: any, aiAnalysis: string, source: string): AnalysisResult {
  // Extract key information from Gemini response
  const isExoplanet = aiAnalysis.toLowerCase().includes('exoplanet') && 
    (aiAnalysis.toLowerCase().includes('yes') || aiAnalysis.toLowerCase().includes('likely'))
  
  // Extract confidence from the response or calculate based on content
  const confidenceMatch = aiAnalysis.match(/(\d+)%/i)
  const confidence = confidenceMatch ? parseInt(confidenceMatch[1]) : 85

  // Extract planet type if mentioned
  const planetTypeMatch = aiAnalysis.match(/(super-earth|gas giant|terrestrial|rocky|ice giant)/i)
  const planetType = planetTypeMatch ? planetTypeMatch[1] : 'Unknown'

  // Create a summary explanation
  const explanation = `Based on Gemini AI analysis, this celestial body ${isExoplanet ? 'is classified as an exoplanet' : 'does not meet exoplanet criteria'}. The AI has provided detailed scientific insights about the planetary characteristics, habitability potential, and scientific significance.`

  // Extract top features from the analysis
  const topFeatures = [
    {
      name: "AI Classification",
      impact: 95,
      value: isExoplanet ? "Exoplanet" : "Not Exoplanet",
      reasoning: "Primary determination from Gemini AI analysis",
    },
    {
      name: "Planet Type",
      impact: 88,
      value: planetType,
      reasoning: "Classification based on physical and orbital characteristics",
    },
    {
      name: "Habitability Assessment",
      impact: 82,
      value: planetData.habitability_score || "Unknown",
      reasoning: "Potential for supporting life based on environmental conditions",
    },
    {
      name: "Orbital Characteristics",
      impact: 76,
      value: `${planetData.orbital_period} days`,
      reasoning: "Orbital period and distance from host star",
    },
    {
      name: "Physical Properties",
      impact: 71,
      value: `${planetData.planet_radius} Earth radii`,
      reasoning: "Size and mass characteristics compared to Earth",
    },
  ]

  return {
    isExoplanet,
    confidence,
    explanation,
    aiAnalysis,
    topFeatures,
    planetData,
    source,
  }
}

// Placeholder AI analysis function
function analyzePlanetData(planetData: any, source: string): AnalysisResult {
  // Extract features for analysis
  const raw = source === "existing" ? planetData.features : planetData
  const features = normalizePlanetData(raw)

  // Simple heuristic: check habitability score or other key metrics
  const habitabilityScore = Number.parseFloat(features.habitability_score) || 0
  const planetRadius = Number.parseFloat(features.planet_radius) || 0
  const orbitalPeriod = Number.parseFloat(features.orbital_period) || 0
  const equilibriumTemp = Number.parseFloat(features.equilibrium_temperature) || 0

  // Determine if it's an exoplanet (placeholder logic)
  const isExoplanet = habitabilityScore > 0.5 || (planetRadius > 0.5 && planetRadius < 10 && orbitalPeriod > 0)

  // Calculate confidence
  const confidence = Math.min(95, Math.max(65, Math.round(habitabilityScore * 100 + Math.random() * 10)))

  // Generate explanation
  const explanation = isExoplanet
    ? `Based on the analysis of ${Object.keys(features).length} planetary features, our AI model has determined this is likely an exoplanet. The combination of orbital characteristics, physical properties, and stellar parameters align with known exoplanet signatures. Key indicators include the planet's radius (${planetRadius} Earth radii), orbital period (${orbitalPeriod} days), and habitability score (${habitabilityScore}). These values fall within the expected range for confirmed exoplanets in our database.`
    : `After analyzing ${Object.keys(features).length} features, our AI model suggests this celestial body does not match typical exoplanet characteristics. The data shows anomalies in key parameters such as orbital mechanics, physical properties, or stellar relationships that deviate from confirmed exoplanet patterns. This could indicate a different type of celestial object, measurement errors, or insufficient data quality.`

  // Determine top impactful features
  const topFeatures = [
    {
      name: "Habitability Score",
      impact: 95,
      value: habitabilityScore.toFixed(2),
      reasoning: "Primary indicator of exoplanet potential based on conditions suitable for life",
    },
    {
      name: "Planet Radius",
      impact: 88,
      value: `${planetRadius} Earth radii`,
      reasoning: "Size comparison to Earth helps classify planet type and formation history",
    },
    {
      name: "Orbital Period",
      impact: 82,
      value: `${orbitalPeriod} days`,
      reasoning: "Determines the planet's year length and distance from its host star",
    },
    {
      name: "Equilibrium Temperature",
      impact: 76,
      value: `${equilibriumTemp} K`,
      reasoning: "Critical for determining potential atmospheric conditions and habitability",
    },
    {
      name: "Semi-Major Axis",
      impact: 71,
      value: `${features.semi_major_axis} AU`,
      reasoning: "Defines orbital distance and influences temperature and stellar radiation received",
    },
  ]

  return {
    isExoplanet,
    confidence,
    explanation,
    topFeatures,
    planetData,
    source,
  }
}

// Generate descriptive query for planet image
function generatePlanetImageQuery(planetData: any): string {
  const features = planetData.features || planetData
  const temp = Number.parseFloat(features.equilibrium_temperature) || 250
  const radius = Number.parseFloat(features.planet_radius) || 1
  const hasRings = features.ring_system === true || features.ring_system === "true"

  let description = "exoplanet in space"

  // Temperature-based appearance
  if (temp < 200) {
    description += " frozen ice planet blue white"
  } else if (temp < 300) {
    description += " earth-like planet blue green"
  } else if (temp < 500) {
    description += " hot desert planet orange red"
  } else {
    description += " lava planet glowing red orange"
  }

  // Size-based
  if (radius > 3) {
    description += " gas giant"
  } else if (radius > 1.5) {
    description += " super earth"
  }

  // Rings
  if (hasRings) {
    description += " with ring system"
  }

  description += " realistic space background stars"

  return description
}

// Normalize new Kepler/KOI field names to previous keys used in analysis
function normalizePlanetData(data: any) {
  if (!data) return {}
  const out: any = { ...data }
  // Map new names -> legacy keys consumed by analysis and image prompt
  if (data.planetary_radius_earth_radii) out.planet_radius = String(data.planetary_radius_earth_radii)
  if (data.orbital_period_days) out.orbital_period = String(data.orbital_period_days)
  if (data.equilibrium_temperature_k) out.equilibrium_temperature = String(data.equilibrium_temperature_k)
  if (data.insolation_flux_earth_flux) out.insolation_flux = String(data.insolation_flux_earth_flux)
  if (data.stellar_effective_temperature_k) out.stellar_temperature = String(data.stellar_effective_temperature_k)
  if (data.stellar_radius_solar_radii) out.stellar_radius = String(data.stellar_radius_solar_radii)
  // Rings not present; default to false
  if (out.ring_system === undefined) out.ring_system = "false"
  return out
}
