"use client"

import { useEffect, useState, Suspense } from "react"
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
  topFeatures: Array<{
    name: string
    impact: number
    value: string | number
    reasoning: string
  }>
  geminiAnalysis?: {
    answer: boolean
    most_important_features: Array<{
      [key: string]: string
      Relevance: string
    }>
  }
  planetData: any
  source: string
}

function ResultsContent() {
  const searchParams = useSearchParams()
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [planetImage, setPlanetImage] = useState<string | null>(null)
  const [isGeneratingImage, setIsGeneratingImage] = useState(false)

  useEffect(() => {
    const planetDataStr = searchParams.get("planetData")
    const analysisResultStr = searchParams.get("analysisResult")
    const source = searchParams.get("source")
    const error = searchParams.get("error")

    if (planetDataStr) {
      const planetData = JSON.parse(planetDataStr)
      
      if (analysisResultStr) {
        // Use real Gemini analysis
        const analysisResult = JSON.parse(analysisResultStr)
        setResult({
          ...analysisResult,
          planetData,
          source: source || "unknown"
        })
      } else {
        // Fallback to mock analysis
        const mockAnalysis = analyzePlanetData(planetData, source || "unknown")
        if (error) {
          mockAnalysis.explanation += ` (${error})`
        }
        setResult(mockAnalysis)
      }
    }
  }, [searchParams])

  // Auto-generate planet image when result is loaded
  useEffect(() => {
    if (result?.planetData && !planetImage && !isGeneratingImage) {
      generatePlanetImage()
    }
  }, [result, planetImage, isGeneratingImage])

  const generatePlanetImage = async () => {
    if (!result?.planetData) return

    setIsGeneratingImage(true)
    try {
      const response = await fetch('/api/generate-planet-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ planetData: result.planetData }),
      })

      if (response.ok) {
        const data = await response.json()
        setPlanetImage(data.imageUrl)
      } else {
        console.error('Failed to generate planet image')
      }
    } catch (error) {
      console.error('Error generating planet image:', error)
    } finally {
      setIsGeneratingImage(false)
    }
  }

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

        {/* AI Analysis and Image Generation */}
        <div className="grid lg:grid-cols-2 gap-6">
          {/* AI Explanation */}
          <Card className="p-6 bg-card/50 backdrop-blur border-2 border-border">
            <h2 className="text-2xl font-bold mb-4 text-primary flex items-center gap-2">
              <TrendingUp className="w-6 h-6" />
              AI Analysis
            </h2>
            <p className="text-foreground leading-relaxed">{result.explanation}</p>
          </Card>

          {/* AI Generated Exoplanet Image */}
          <Card className="p-6 bg-card/50 backdrop-blur border-2 border-border">
            <h2 className="text-2xl font-bold mb-4 text-primary flex items-center gap-2">
              <span className="text-2xl">🪐</span>
              AI Generated Exoplanet
            </h2>
            <div className="space-y-4">
              {planetImage ? (
                <div className="relative">
                  <img
                    src={planetImage}
                    alt="AI Generated Exoplanet Visualization"
                    className="w-full h-64 object-cover rounded-lg border border-border/50"
                  />
                  <div className="absolute top-2 left-2 bg-black/50 text-white px-2 py-1 rounded text-sm">
                    AI Generated
                  </div>
                  <div className="absolute top-2 right-2">
                    <Button
                      onClick={generatePlanetImage}
                      disabled={isGeneratingImage}
                      size="sm"
                      variant="outline"
                      className="bg-black/50 text-white border-white/20 hover:bg-black/70"
                    >
                      {isGeneratingImage ? (
                        <>
                          <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                          Regenerating...
                        </>
                      ) : (
                        <>
                          🔄 Regenerate
                        </>
                      )}
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="w-full h-64 border-2 border-dashed border-border/50 rounded-lg flex items-center justify-center bg-background/30">
                  <div className="text-center">
                    {isGeneratingImage ? (
                      <>
                        <div className="text-4xl mb-4 animate-pulse">🪐</div>
                        <p className="text-muted-foreground mb-4">Generating AI visualization...</p>
                        <div className="flex items-center justify-center gap-2">
                          <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                          <span className="text-sm text-muted-foreground">This may take a moment</span>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="text-4xl mb-4">🪐</div>
                        <p className="text-muted-foreground mb-4">AI-generated visualization based on your data</p>
                        <Button
                          onClick={generatePlanetImage}
                          disabled={isGeneratingImage}
                          className="gap-2"
                        >
                          {isGeneratingImage ? (
                            <>
                              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                              Generating...
                            </>
                          ) : (
                            <>
                              Generate Planet Image
                            </>
                          )}
                        </Button>
                      </>
                    )}
                  </div>
                </div>
              )}
              <p className="text-sm text-muted-foreground text-center">
                AI-generated visualization based on Kepler dataset parameters
              </p>
            </div>
          </Card>
        </div>

        {/* Feature Impact Analysis */}
        <Card className="p-6 mt-8 bg-card/50 backdrop-blur border-2 border-border">
          <h2 className="text-2xl font-bold mb-6 text-primary">Key Features Impact</h2>
          <div className="space-y-6">
            {result.geminiAnalysis?.most_important_features ?
              result.geminiAnalysis.most_important_features.map((feature, index) => {
                console.log('Rendering Gemini feature:', feature)
                // Extract feature name and description
                const featureName = Object.keys(feature).find(key => key !== 'Relevance')
                const featureDescription = featureName ? feature[featureName] : ''
                const relevance = feature.Relevance

                // Determine relevance color based on text content
                const getRelevanceColor = (relevance: string) => {
                  if (relevance.toLowerCase().includes('high') || relevance.toLowerCase().includes('critical') || relevance.toLowerCase().includes('very high')) {
                    return 'text-red-400'
                  } else if (relevance.toLowerCase().includes('medium') || relevance.toLowerCase().includes('moderate')) {
                    return 'text-yellow-400'
                  } else {
                    return 'text-green-400'
                  }
                }

                // Format feature name from 'feature1' to 'Feature 1'
                const formattedFeatureName = featureName ? featureName.replace(/^feature(\d+)$/, 'Feature $1') : '';

                return (
                  <div key={index} className="space-y-3 p-4 bg-background/30 rounded-lg border border-border/50">
                    <div className="flex items-center gap-3">
                      <Badge variant="outline" className="text-lg px-3 py-1">
                        #{index + 1}
                      </Badge>
                      <span className="font-semibold text-foreground">{formattedFeatureName}</span>
                    </div>

                    <div className="pl-14 space-y-2">
                      <div>
                        <p className="text-sm text-muted-foreground mb-1">Feature & Explanation:</p>
                        <p className="text-sm text-foreground">{featureDescription}</p>
                      </div>

                      <div>
                        <p className="text-sm text-muted-foreground mb-1">Relevance:</p>
                        <p className={`text-sm font-medium ${getRelevanceColor(relevance)}`}>{relevance}</p>
                      </div>
                    </div>
                  </div>
                )
              }) : result.topFeatures.map((feature, index) => (
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

// Placeholder AI analysis function
function analyzePlanetData(planetData: any, source: string): AnalysisResult {
  // Extract features for analysis
  const features = source === "existing" ? planetData.features : planetData

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

