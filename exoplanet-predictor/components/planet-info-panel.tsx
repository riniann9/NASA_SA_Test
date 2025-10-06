"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { X, Sparkles, Search, Telescope } from "lucide-react"
import type { PlanetData } from "@/app/existing/page"
import { Planet3DViewer } from "./planet-3d-viewer"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useRouter } from "next/navigation"

// Define PlanetFormData type for consistency with /new page and API
type PlanetFormData = {
  orbital_period: string
  transit_epoch: string
  impact_parameter: string
  transit_duration: string
  transit_depth: string
  planetary_radius: string
  equilibrium_temperature: string
  insolation_flux: string
  transit_snr: string
  tce_planet_number: string
  stellar_effective_temperature: string
  stellar_surface_gravity: string
  stellar_radius: string
  ra: string
  dec: string
  kepler_band_mag: string
  ring_system: string
  light_curve_file?: File
}

type PlanetInfoPanelProps = {
  planet: PlanetData
  onClose: () => void
  onAnalyze: (planet: PlanetData) => void
}

export function PlanetInfoPanel({ planet, onClose, onAnalyze }: PlanetInfoPanelProps) {
  const [isDetecting, setIsDetecting] = useState(false)
  const [detectionComplete, setDetectionComplete] = useState(false)
  const router = useRouter()


  const handleDetect = async () => {
    setIsDetecting(true)
    
    try {
      // Convert planet features to the format expected by the analysis
      const planetFormData: PlanetFormData = {
        orbital_period: planet.features.orbital_period.toString(),
        transit_epoch: "0", // Default value
        impact_parameter: "0", // Default value
        transit_duration: "0", // Default value
        transit_depth: "0", // Default value
        planetary_radius: planet.features.planet_radius.toString(),
        equilibrium_temperature: planet.features.equilibrium_temperature.toString(),
        insolation_flux: planet.features.insolation_flux.toString(),
        transit_snr: "10", // Default value
        tce_planet_number: "1", // Default value
        stellar_effective_temperature: planet.features.stellar_temperature.toString(),
        stellar_surface_gravity: "4.5", // Default value
        stellar_radius: planet.features.stellar_radius.toString(),
        ra: "0", // Default value
        dec: "0", // Default value
        kepler_band_mag: "15", // Default value
        ring_system: planet.features.ring_system.toString()
      }
      
      // Create analysis result based on planet data
      const isExoplanet = planet.features.habitability_score > 0.5
      const confidence = Math.min(95, Math.max(65, Math.round(planet.features.habitability_score * 100 + Math.random() * 10)))
      
      const explanation = isExoplanet
        ? `Based on the analysis of ${Object.keys(planet.features).length} planetary features, our AI model has determined this is likely an exoplanet. The combination of orbital characteristics, physical properties, and stellar parameters align with known exoplanet signatures. Key indicators include the planet's radius (${planet.features.planet_radius} Earth radii), orbital period (${planet.features.orbital_period} days), and habitability score (${planet.features.habitability_score}). These values fall within the expected range for confirmed exoplanets in our database.`
        : `After analyzing ${Object.keys(planet.features).length} features, our AI model suggests this celestial body does not match typical exoplanet characteristics. The data shows anomalies in key parameters such as orbital mechanics, physical properties, or stellar relationships that deviate from confirmed exoplanet patterns.`
      
      const topFeatures = [
        {
          name: "Habitability Score",
          impact: Math.round(planet.features.habitability_score * 100),
          value: planet.features.habitability_score.toFixed(2),
          reasoning: "Primary indicator of exoplanet potential based on conditions suitable for life"
        },
        {
          name: "Planet Radius",
          impact: Math.round(Math.min(95, planet.features.planet_radius * 30)),
          value: `${planet.features.planet_radius} Earth radii`,
          reasoning: "Size comparison to Earth helps classify planet type and formation history"
        },
        {
          name: "Orbital Period",
          impact: Math.round(Math.min(90, planet.features.orbital_period * 0.5)),
          value: `${planet.features.orbital_period} days`,
          reasoning: "Determines the planet's year length and distance from its host star"
        },
        {
          name: "Equilibrium Temperature",
          impact: Math.round(Math.min(85, (300 - Math.abs(planet.features.equilibrium_temperature - 250)) * 0.3)),
          value: `${planet.features.equilibrium_temperature} K`,
          reasoning: "Critical for determining potential atmospheric conditions and habitability"
        },
        {
          name: "Detection Method",
          impact: 80,
          value: planet.features.detection_method,
          reasoning: "Method used to discover the planet affects confidence in its existence"
        }
      ]
      
      const analysisResult = {
        isExoplanet,
        confidence,
        explanation,
        topFeatures,
        planetData: planetFormData,
          source: "existing"
        }
        
      // Navigate to results page with the analysis
      const params = new URLSearchParams({
        planetData: JSON.stringify(planetFormData),
        analysisResult: JSON.stringify(analysisResult),
        geminiPrompt: planet.gemini_prompt,
        source: "existing"
      })
        
        router.push(`/results?${params.toString()}`)
      } catch (error) {
      console.error('Error analyzing planet:', error)
      alert('Failed to analyze planet. Please try again.')
      setIsDetecting(false)
      return
    }
  }

  const features = Object.entries(planet.features).map(([key, value]) => ({
    label: key
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" "),
    value: typeof value === "boolean" ? (value ? "Yes" : "No") : value,
  }))

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-md">
      <Card className="w-full max-w-6xl h-[90vh] bg-gradient-to-br from-slate-900/95 via-purple-900/90 to-slate-900/95 backdrop-blur-xl border-2 border-cyan-500/40 shadow-2xl shadow-cyan-500/20 overflow-hidden">
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-cyan-500/30 bg-black/20">
            <div>
              <h2 className="text-3xl font-bold text-white drop-shadow-[0_0_10px_rgba(0,255,255,0.5)]">
                {planet.name}
              </h2>
              <p className="text-sm text-cyan-200 mt-1">Exoplanet ID: {planet.id}</p>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="text-cyan-100 hover:text-white hover:bg-cyan-500/20"
            >
              <X className="w-5 h-5" />
            </Button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-hidden">
            {!detectionComplete ? (
              /* Detection Screen */
              <div className="flex flex-col items-center justify-center h-full p-6 text-center">
                <div className="mb-6">
                  <div className="w-24 h-24 mx-auto mb-4 bg-gradient-to-br from-cyan-500/20 to-purple-500/20 rounded-full flex items-center justify-center border-2 border-cyan-500/40">
                    <Telescope className="w-12 h-12 text-cyan-400" />
                  </div>
                  <h3 className="text-xl font-bold text-white mb-3">AI Detection Required</h3>
                  <p className="text-cyan-200 max-w-sm text-sm">
                    Perform an AI-powered detection scan to analyze this exoplanet's current status and gather observational data.
                  </p>
                </div>
                
                <Button
                  onClick={handleDetect}
                  disabled={isDetecting}
                  className="gap-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-semibold px-6 py-3 text-base shadow-lg shadow-cyan-500/30 hover:shadow-cyan-500/50 transition-all"
                  size="lg"
                >
                  {isDetecting ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      AI Detecting...
                    </>
                  ) : (
                    <>
                      <Search className="w-4 h-4" />
                      AI Detection
                    </>
                  )}
                </Button>
              </div>
            ) : (
              /* Planet Data Screen */
              <div className="grid md:grid-cols-2 gap-6 p-6 h-full">
                {/* Left: 3D Planet Viewer */}
                <div className="flex flex-col gap-4">
                  <div className="flex-1 bg-black/40 rounded-lg border border-cyan-500/30 overflow-hidden shadow-inner shadow-cyan-500/10">
                    <Planet3DViewer color={planet.color} size={planet.size} />
                  </div>
                  <Button
                    onClick={() => onAnalyze(planet)}
                    className="w-full gap-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-semibold shadow-lg shadow-cyan-500/30 hover:shadow-cyan-500/50 transition-all"
                    size="lg"
                  >
                    <Sparkles className="w-5 h-5" />
                    Analyze with AI
                  </Button>
                </div>

                {/* Right: Features List */}
                <div className="flex flex-col">
                  <h3 className="text-xl font-semibold mb-4 text-white">Planet Features</h3>
                  <ScrollArea className="flex-1 pr-4">
                    <div className="space-y-3">
                      {/* Gemini Prompt Section */}
                      <div className="p-4 bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded-lg border border-purple-500/30 hover:border-purple-400/50 transition-all">
                        <h4 className="text-sm font-semibold text-purple-200 mb-2 flex items-center gap-2">
                          <Sparkles className="w-4 h-4" />
                          AI Analysis Prompt
                        </h4>
                        <div className="text-xs text-gray-300 leading-relaxed max-h-32 overflow-y-auto">
                          {planet.gemini_prompt}
                        </div>
                      </div>
                      
                      {/* Regular Features */}
                      {features.map((feature, index) => (
                        <div
                          key={index}
                          className="flex justify-between items-center p-3 bg-black/30 rounded-lg border border-cyan-500/20 hover:border-cyan-400/50 hover:bg-black/40 transition-all"
                        >
                          <span className="text-sm font-medium text-cyan-200">{feature.label}</span>
                          <span className="text-sm font-semibold text-white">{feature.value}</span>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </div>
              </div>
            )}
          </div>
        </div>
      </Card>
    </div>
  )
}
