"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ArrowLeft, Sparkles, Rocket } from "lucide-react"
import Link from "next/link"
import { LoadingScreen } from "@/components/loading-screen"

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

export default function NewPage() {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isAnalyzingLightCurve, setIsAnalyzingLightCurve] = useState(false)
  const [lightCurveResults, setLightCurveResults] = useState<any>(null)
  const [formData, setFormData] = useState<PlanetFormData>({
    orbital_period: "",
    transit_epoch: "",
    impact_parameter: "",
    transit_duration: "",
    transit_depth: "",
    planetary_radius: "",
    equilibrium_temperature: "",
    insolation_flux: "",
    transit_snr: "",
    tce_planet_number: "",
    stellar_effective_temperature: "",
    stellar_surface_gravity: "",
    stellar_radius: "",
    ra: "",
    dec: "",
    kepler_band_mag: "",
    ring_system: "",
  })

  const handleInputChange = (field: keyof PlanetFormData, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  const handleFileChange = (file: File | null) => {
    setFormData((prev) => ({ ...prev, light_curve_file: file || undefined }))
    setLightCurveResults(null) // Clear previous results when new file is selected
  }

  const handleLightCurveAnalysis = async () => {
    if (!formData.light_curve_file) {
      alert('Please select a FITS file first')
      return
    }

    setIsAnalyzingLightCurve(true)
    try {
      const formDataToSend = new FormData()
      formDataToSend.append('fits_file', formData.light_curve_file)

      const response = await fetch('/api/analyze-light-curve', {
        method: 'POST',
        body: formDataToSend,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Failed to analyze light curve')
      }

      const results = await response.json()
      setLightCurveResults(results)
    } catch (error) {
      console.error('Error analyzing light curve:', error)
      alert(`Failed to analyze light curve: ${error instanceof Error ? error.message : 'Unknown error'}. Make sure the Python server is running on port 5000.`)
    } finally {
      setIsAnalyzingLightCurve(false)
    }
  }

  const handlePrefillExample = () => {
    setFormData({
      orbital_period: "9.48803557",
      transit_epoch: "170.53875",
      impact_parameter: "0.146",
      transit_duration: "2.9575",
      transit_depth: "615.8",
      planetary_radius: "2.26",
      equilibrium_temperature: "793",
      insolation_flux: "93.59",
      transit_snr: "35.80",
      tce_planet_number: "1",
      stellar_effective_temperature: "5455",
      stellar_surface_gravity: "4.467",
      stellar_radius: "0.927",
      ra: "291.934230",
      dec: "48.141651",
      kepler_band_mag: "15.347",
      ring_system: "false",
    })
    setLightCurveResults(null) // Clear any previous light curve results
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsAnalyzing(true)

    try {
      // Call Gemini API for real analysis
      const response = await fetch('/api/analyze-planet', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          planetData: formData
        })
      })

      if (!response.ok) {
        throw new Error('Failed to analyze planet data')
      }

      const analysisResult = await response.json()

      // Navigate to results with analysis data
      const params = new URLSearchParams({
        planetData: JSON.stringify(formData),
        analysisResult: JSON.stringify(analysisResult),
        source: "new",
      })
      window.location.href = `/results?${params.toString()}`
    } catch (error) {
      console.error('Error analyzing planet:', error)
      // Fallback to mock analysis if API fails
      await new Promise((resolve) => setTimeout(resolve, 2000))
      
      const params = new URLSearchParams({
        planetData: JSON.stringify(formData),
        source: "new",
        error: "API analysis failed, showing mock results"
      })
      window.location.href = `/results?${params.toString()}`
    }
  }

  if (isAnalyzing) {
    return <LoadingScreen />
  }

  const inputSections = [
    {
      title: "Orbital & Transit Properties",
      icon: "🔄",
      fields: [
        { key: "orbital_period", label: "Orbital Period [days]", type: "number", placeholder: "e.g., 112.3" },
        { key: "transit_epoch", label: "Transit Epoch [BKJD]", type: "number", placeholder: "e.g., 2454833.0" },
        { key: "impact_parameter", label: "Impact Parameter", type: "number", placeholder: "e.g., 0.1" },
        { key: "transit_duration", label: "Transit Duration [hrs]", type: "number", placeholder: "e.g., 2.5" },
        { key: "transit_depth", label: "Transit Depth [ppm]", type: "number", placeholder: "e.g., 150" },
      ],
    },
    {
      title: "Planetary Properties",
      icon: "🌍",
      fields: [
        { key: "planetary_radius", label: "Planetary Radius [Earth radii]", type: "number", placeholder: "e.g., 1.34" },
        { key: "equilibrium_temperature", label: "Equilibrium Temperature [K]", type: "number", placeholder: "e.g., 233" },
        { key: "insolation_flux", label: "Insolation Flux [Earth flux]", type: "text", placeholder: "e.g., 93.59 (+29.45/-16.65)" },
      ],
    },
    {
      title: "Detection & Signal Properties",
      icon: "📡",
      fields: [
        { key: "transit_snr", label: "Transit Signal-to-Noise", type: "number", placeholder: "e.g., 35.80" },
        { key: "tce_planet_number", label: "TCE Planet Number", type: "number", placeholder: "e.g., 1" },
      ],
    },
    {
      title: "Stellar Properties",
      icon: "⭐",
      fields: [
        { key: "stellar_effective_temperature", label: "Stellar Effective Temperature [K]", type: "number", placeholder: "e.g., 4402" },
        { key: "stellar_surface_gravity", label: "Stellar Surface Gravity [log10(cm/s**2)]", type: "number", placeholder: "e.g., 4.5" },
        { key: "stellar_radius", label: "Stellar Radius [Solar radii]", type: "number", placeholder: "e.g., 0.6" },
      ],
    },
    {
      title: "Observational Properties",
      icon: "🔭",
      fields: [
        { key: "ra", label: "RA [decimal degrees]", type: "number", placeholder: "e.g., 285.0" },
        { key: "dec", label: "Dec [decimal degrees]", type: "number", placeholder: "e.g., 45.0" },
        { key: "kepler_band_mag", label: "Kepler-band [mag]", type: "number", placeholder: "e.g., 12.5" },
      ],
    },
  ]

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Cosmic background effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 right-1/4 w-96 h-96 bg-primary/10 rounded-full blur-[120px] animate-pulse" />
        <div
          className="absolute bottom-1/4 left-1/4 w-96 h-96 bg-accent/10 rounded-full blur-[120px] animate-pulse"
          style={{ animationDelay: "1s" }}
        />
      </div>

      <div className="relative z-10 container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <Link href="/">
            <Button variant="outline" size="sm" className="gap-2 bg-transparent">
              <ArrowLeft className="w-4 h-4" />
              Back to Home
            </Button>
          </Link>
        </div>

        {/* Title */}
        <div className="text-center mb-12 space-y-4">
          <div className="flex items-center justify-center gap-3">
            <Rocket className="w-12 h-12 text-primary animate-float" />
            <h1 className="text-5xl font-bold text-primary">New Planet Prediction</h1>
          </div>
          <p className="text-xl text-muted-foreground">
            Enter the features of your celestial body to determine if it's an exoplanet
          </p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit}>
          <div className="grid lg:grid-cols-2 gap-6 mb-8">
            {inputSections.map((section, sectionIndex) => (
              <Card
                key={sectionIndex}
                className="p-6 bg-card/50 backdrop-blur border-2 border-border hover:border-primary/50 transition-colors"
              >
                <div className="flex items-center gap-3 mb-6">
                  <span className="text-3xl">{section.icon}</span>
                  <h2 className="text-2xl font-bold text-foreground">{section.title}</h2>
                </div>

                <div className="space-y-4">
                  {section.fields.map((field) => (
                    <div key={field.key} className="space-y-2">
                      <Label htmlFor={field.key} className="text-sm font-medium text-foreground">
                        {field.label}
                      </Label>
                      <Input
                        id={field.key}
                        type={field.type}
                        placeholder={field.placeholder}
                        value={formData[field.key as keyof PlanetFormData]}
                        onChange={(e) => handleInputChange(field.key as keyof PlanetFormData, e.target.value)}
                        className="bg-background/50 border-border focus:border-primary"
                        required
                      />
                    </div>
                  ))}
                </div>
              </Card>
            ))}

            {/* Ring System Card */}
            <Card className="p-6 bg-card/50 backdrop-blur border-2 border-border hover:border-primary/50 transition-colors">
              <div className="flex items-center gap-3 mb-6">
                <span className="text-3xl">💍</span>
                <h2 className="text-2xl font-bold text-foreground">Ring System</h2>
              </div>

              <div className="space-y-2">
                <Label htmlFor="ring_system" className="text-sm font-medium text-foreground">
                  Does the planet have a ring system?
                </Label>
                <Select value={formData.ring_system} onValueChange={(value) => handleInputChange("ring_system", value)}>
                  <SelectTrigger className="bg-background/50 border-border focus:border-primary">
                    <SelectValue placeholder="Select option" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="false">No</SelectItem>
                    <SelectItem value="true">Yes</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </Card>

            {/* Light Curve Analysis Card */}
            <Card className="p-6 bg-card/50 backdrop-blur border-2 border-border hover:border-primary/50 transition-colors">
              <div className="flex items-center gap-3 mb-6">
                <span className="text-3xl">📊</span>
                <h2 className="text-2xl font-bold text-foreground">Light Curve Analysis</h2>
              </div>

              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="light_curve_file" className="text-sm font-medium text-foreground">
                    Upload FITS File (Optional)
                  </Label>
                  <Input
                    id="light_curve_file"
                    type="file"
                    accept=".fits,.fit"
                    onChange={(e) => handleFileChange(e.target.files?.[0] || null)}
                    className="bg-background/50 border-border focus:border-primary"
                  />
                  <p className="text-xs text-muted-foreground">
                    Upload a FITS file to analyze light curve features
                  </p>
                </div>

                {formData.light_curve_file && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-foreground">
                        Selected: {formData.light_curve_file.name}
                      </span>
                    </div>
                    
                    <Button
                      type="button"
                      onClick={handleLightCurveAnalysis}
                      disabled={isAnalyzingLightCurve}
                      className="w-full bg-cyan-600 hover:bg-cyan-700 text-white"
                    >
                      {isAnalyzingLightCurve ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                          Analyzing Light Curve...
                        </>
                      ) : (
                        <>
                          <span className="text-lg mr-2">🔬</span>
                          Analyze Light Curve Features
                        </>
                      )}
                    </Button>
                  </div>
                )}

                {lightCurveResults && (
                  <div className="mt-4 p-4 bg-background/30 rounded-lg border border-border/50">
                    <h3 className="text-lg font-semibold text-foreground mb-3">Light Curve Analysis Results</h3>
                    <div className="space-y-4 max-h-80 overflow-y-auto">
                      {/* Gemini Prompt Section */}
                      {lightCurveResults.gemini_prompt_human && (
                        <div className="space-y-2">
                          <h4 className="text-md font-semibold text-cyan-400">Analysis Prompt:</h4>
                          <div className="p-3 bg-background/50 rounded border border-cyan-400/20">
                            <p className="text-sm text-foreground leading-relaxed">
                              {lightCurveResults.gemini_prompt_human}
                            </p>
                          </div>
                        </div>
                      )}

                      {/* Random Forest Prediction Section */}
                      {lightCurveResults.random_forest_prediction && (
                        <div className="space-y-2">
                          <h4 className="text-md font-semibold text-green-400">🤖 Random Forest Prediction:</h4>
                          <div className="p-3 bg-background/50 rounded border border-green-400/20">
                            <div className="space-y-1">
                              <p className="text-sm">
                                <span className="font-semibold text-green-400">Prediction:</span> 
                                <span className={`ml-2 font-bold ${
                                  lightCurveResults.random_forest_prediction.prediction === 'EXOPLANET' 
                                    ? 'text-green-400' 
                                    : 'text-red-400'
                                }`}>
                                  {lightCurveResults.random_forest_prediction.prediction}
                                </span>
                              </p>
                              <p className="text-sm">
                                <span className="font-semibold text-green-400">Confidence:</span> 
                                <span className="ml-2 text-foreground">
                                  {lightCurveResults.random_forest_prediction.confidence}
                                </span>
                              </p>
                              <p className="text-sm">
                                <span className="font-semibold text-green-400">Probability:</span> 
                                <span className="ml-2 text-foreground">
                                  {(lightCurveResults.random_forest_prediction.probability * 100).toFixed(1)}%
                                </span>
                              </p>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Features Section */}
                      {lightCurveResults.Features && (
                        <div className="space-y-2">
                          <h4 className="text-md font-semibold text-blue-400">📊 Features (labeled):</h4>
                          <div className="p-3 bg-background/50 rounded border border-blue-400/20">
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
                              {Object.entries(lightCurveResults.Features).map(([key, value]) => (
                                <div key={key} className="flex justify-between">
                                  <span className="text-muted-foreground font-mono">{key}:</span>
                                  <span className="text-foreground font-mono">{String(value)}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Fallback for other results */}
                      {!lightCurveResults.gemini_prompt_human && !lightCurveResults.random_forest_prediction && !lightCurveResults.Features && (
                        <div className="space-y-2">
                          <h4 className="text-md font-semibold text-foreground">Raw Analysis Results:</h4>
                          <div className="p-3 bg-background/50 rounded border border-border/50">
                            <div className="space-y-1 max-h-40 overflow-y-auto">
                              {Object.entries(lightCurveResults).map(([key, value]) => (
                                <div key={key} className="flex justify-between text-sm">
                                  <span className="text-muted-foreground font-mono">{key}:</span>
                                  <span className="text-foreground font-mono">{String(value)}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </Card>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button
              type="button"
              onClick={handlePrefillExample}
              size="lg"
              variant="outline"
              className="gap-3 bg-transparent border-green-400 text-green-400 hover:bg-green-400/10 px-8 py-6 text-lg h-auto"
            >
              <span className="text-lg">📋</span>
              Prefill Example Data
            </Button>
            <Button
              type="submit"
              size="lg"
              className="gap-3 bg-primary hover:bg-primary/90 text-primary-foreground px-12 py-6 text-lg h-auto group"
            >
              <Sparkles className="w-6 h-6 group-hover:animate-spin" />
              Analyze with AI
              <Sparkles className="w-6 h-6 group-hover:animate-spin" />
            </Button>
            <Link href="http://192.168.1.94:5000/" target="_blank" rel="noopener noreferrer">
              <Button
                type="button"
                size="lg"
                variant="outline"
                className="gap-3 bg-transparent border-cyan-400 text-cyan-400 hover:bg-cyan-400/10 px-12 py-6 text-lg h-auto"
              >
                <span className="text-lg">📊</span>
                Analyze Light Curve
              </Button>
            </Link>
          </div>
        </form>
      </div>
    </div>
  )
}
