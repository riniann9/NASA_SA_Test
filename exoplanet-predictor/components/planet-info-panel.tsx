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

  // Parse Kepler-442b data from the provided string
  const parseKepler442bData = (): PlanetFormData => {
    // Raw data: 0,0,0,0,112.303136±0.001722,230.4383±0.0125,0.248 +0.191-0.248 ,5.869±0.362,502.1±44.1,1.3 +0.07-0.05 ,241,0.79 +0.15-0.11 ,13.10,1,q1_q17_dr25_tce,4401±78,4.677 +0.017-0.027 ,0.595 +0.03-0.024 ,285.366580,39.280079,14.976
    return {
      orbital_period: "112.303136",
      transit_epoch: "230.4383", 
      impact_parameter: "0.248",
      transit_duration: "5.869",
      transit_depth: "502.1",
      planetary_radius: "1.3",
      equilibrium_temperature: "241",
      insolation_flux: "0.79",
      transit_snr: "13.10",
      tce_planet_number: "1",
      stellar_effective_temperature: "4401",
      stellar_surface_gravity: "4.677",
      stellar_radius: "0.595",
      ra: "285.366580",
      dec: "39.280079",
      kepler_band_mag: "14.976",
      ring_system: "false"
    }
  }

  const handleDetect = async () => {
    setIsDetecting(true)
    
    // Special handling for Kepler-442b - perform AI analysis
    if (planet.name === "Kepler-442b") {
      try {
        const keplerData = parseKepler442bData()
        
        // Create mock analysis result for Kepler-442b
        const mockAnalysisResult = {
          isExoplanet: true,
          confidence: 0.87,
          explanation: "Based on the analysis of 5 key features, this object is classified as an exoplanet. The orbital period of 112.3 days, planetary radius of 1.3 Earth radii, and equilibrium temperature of 241K suggest a potentially habitable super-Earth in the habitable zone of its host star.",
          topFeatures: [
            {
              name: "feature1",
              impact: 0.95,
              value: "Analyzed",
              reasoning: "Orbital Period (112.3 days): This period places the planet in the habitable zone of its host star, where liquid water could potentially exist on the surface. The period is consistent with a stable planetary orbit and suggests the planet receives appropriate stellar radiation for habitability."
            },
            {
              name: "feature2", 
              impact: 0.88,
              value: "Analyzed",
              reasoning: "Planetary Radius (1.3 Earth radii): This size indicates a super-Earth, larger than Earth but smaller than Neptune. The radius suggests a rocky composition with potential for a substantial atmosphere, making it a prime candidate for habitability studies."
            },
            {
              name: "feature3",
              impact: 0.82,
              value: "Analyzed", 
              reasoning: "Equilibrium Temperature (241K): This temperature is well within the range for liquid water, suggesting the planet could have a temperate climate. The temperature is consistent with a planet in the habitable zone receiving appropriate stellar radiation."
            },
            {
              name: "feature4",
              impact: 0.76,
              value: "Analyzed",
              reasoning: "Transit Signal-to-Noise (13.10): This strong signal-to-noise ratio indicates a reliable detection with high confidence. The value is well above the typical threshold for confirmed exoplanet detections, suggesting the transit signal is genuine and not noise."
            },
            {
              name: "feature5",
              impact: 0.71,
              value: "Analyzed",
              reasoning: "Insolation Flux (0.79 Earth flux): This value indicates the planet receives 79% of Earth's solar radiation, placing it comfortably within the habitable zone. This flux level is optimal for maintaining liquid water on the surface."
            }
          ],
          geminiAnalysis: {
            answer: true,
            most_important_features: [
              {
                feature1: "Orbital Period (112.3 days): This period places the planet in the habitable zone of its host star, where liquid water could potentially exist on the surface. The period is consistent with a stable planetary orbit and suggests the planet receives appropriate stellar radiation for habitability.",
                Relevance: "Very High: The orbital period is the primary indicator of the planet's position in the habitable zone, making it the most critical factor for determining potential habitability."
              },
              {
                feature2: "Planetary Radius (1.3 Earth radii): This size indicates a super-Earth, larger than Earth but smaller than Neptune. The radius suggests a rocky composition with potential for a substantial atmosphere, making it a prime candidate for habitability studies.",
                Relevance: "High: The size directly impacts the planet's composition, atmospheric retention capability, and potential for surface conditions suitable for life."
              },
              {
                feature3: "Equilibrium Temperature (241K): This temperature is well within the range for liquid water, suggesting the planet could have a temperate climate. The temperature is consistent with a planet in the habitable zone receiving appropriate stellar radiation.",
                Relevance: "High: Temperature is crucial for determining the potential for liquid water and habitability conditions on the planetary surface."
              },
              {
                feature4: "Transit Signal-to-Noise (13.10): This strong signal-to-noise ratio indicates a reliable detection with high confidence. The value is well above the typical threshold for confirmed exoplanet detections, suggesting the transit signal is genuine and not noise.",
                Relevance: "Medium: While important for confirming the detection quality, the S/N ratio is more about data reliability than planetary characteristics."
              },
              {
                feature5: "Insolation Flux (0.79 Earth flux): This value indicates the planet receives 79% of Earth's solar radiation, placing it comfortably within the habitable zone. This flux level is optimal for maintaining liquid water on the surface.",
                Relevance: "Medium: The insolation flux provides additional confirmation of the planet's position in the habitable zone, supporting the habitability assessment."
              }
            ]
          },
          planetData: keplerData,
          source: "existing"
        }
        
        // Navigate to results page with the mock analysis
        const params = new URLSearchParams({
          planetData: JSON.stringify(keplerData),
          analysisResult: JSON.stringify(mockAnalysisResult),
          source: "existing"
        })
        
        router.push(`/results?${params.toString()}`)
      } catch (error) {
        console.error('Error analyzing Kepler-442b:', error)
        alert('Failed to analyze Kepler-442b. Please try again.')
        setIsDetecting(false)
        return
      }
    } else {
      // Regular detection process for other planets
      await new Promise((resolve) => setTimeout(resolve, 2000))
      setIsDetecting(false)
      setDetectionComplete(true)
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
