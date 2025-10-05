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
}

export default function NewPage() {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
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
  })

  const handleInputChange = (field: keyof PlanetFormData, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
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
          </div>

          {/* Submit Button */}
          <div className="flex justify-center">
            <Button
              type="submit"
              size="lg"
              className="gap-3 bg-primary hover:bg-primary/90 text-primary-foreground px-12 py-6 text-lg h-auto group"
            >
              <Sparkles className="w-6 h-6 group-hover:animate-spin" />
              Analyze with AI
              <Sparkles className="w-6 h-6 group-hover:animate-spin" />
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}
