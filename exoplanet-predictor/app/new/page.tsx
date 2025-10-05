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
  kepid: string
  koi_name: string
  kepler_name: string
  orbital_period_days: string
  transit_epoch_bkjd: string
  impact_parameter: string
  transit_duration_hrs: string
  transit_depth_ppm: string
  planetary_radius_earth_radii: string
  equilibrium_temperature_k: string
  insolation_flux_earth_flux: string
  transit_snr: string
  tce_planet_number: string
  tce_delivery: string
  stellar_effective_temperature_k: string
  stellar_surface_gravity_log: string
  stellar_radius_solar_radii: string
  ra_deg: string
  dec_deg: string
  kepler_mag: string
}

export default function NewPage() {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [formData, setFormData] = useState<PlanetFormData>({
    kepid: "",
    koi_name: "",
    kepler_name: "",
    orbital_period_days: "",
    transit_epoch_bkjd: "",
    impact_parameter: "",
    transit_duration_hrs: "",
    transit_depth_ppm: "",
    planetary_radius_earth_radii: "",
    equilibrium_temperature_k: "",
    insolation_flux_earth_flux: "",
    transit_snr: "",
    tce_planet_number: "",
    tce_delivery: "",
    stellar_effective_temperature_k: "",
    stellar_surface_gravity_log: "",
    stellar_radius_solar_radii: "",
    ra_deg: "",
    dec_deg: "",
    kepler_mag: "",
  })

  const handleInputChange = (field: keyof PlanetFormData, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  const populateSampleData = () => {
    const sampleData: PlanetFormData = {
      kepid: "10797460",
      koi_name: "K00752.01",
      kepler_name: "Kepler-227b",
      orbital_period_days: "9.48803557±2.775e-05",
      transit_epoch_bkjd: "170.53875±0.00216",
      impact_parameter: "0.146 (+0.318/-0.146)",
      transit_duration_hrs: "2.9575±0.0819",
      transit_depth_ppm: "615.8±19.5",
      planetary_radius_earth_radii: "2.26 (+0.26/-0.15)",
      equilibrium_temperature_k: "793",
      insolation_flux_earth_flux: "93.59 (+29.45/-16.65)",
      transit_snr: "35.80",
      tce_planet_number: "1",
      tce_delivery: "q1_q17_dr25_tce",
      stellar_effective_temperature_k: "5455±81",
      stellar_surface_gravity_log: "4.467 (+0.064/-0.096)",
      stellar_radius_solar_radii: "0.927 (+0.105/-0.061)",
      ra_deg: "291.934230",
      dec_deg: "48.141651",
      kepler_mag: "15.347",
    }
    setFormData(sampleData)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsAnalyzing(true)

    try {
      // Call Gemini API for analysis
      const response = await fetch('/api/analyze-planet', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ formData }),
      })

      if (!response.ok) {
        throw new Error('Failed to analyze planet data')
      }

      const result = await response.json()
      
      // Navigate to results with form data and AI analysis
      const params = new URLSearchParams({
        planetData: JSON.stringify(formData),
        aiAnalysis: JSON.stringify(result.analysis),
        source: "new",
      })
      window.location.href = `/results?${params.toString()}`
    } catch (error) {
      console.error('Error analyzing planet:', error)
      // Fallback to results without AI analysis
      const params = new URLSearchParams({
        planetData: JSON.stringify(formData),
        source: "new",
        error: "AI analysis failed"
      })
      window.location.href = `/results?${params.toString()}`
    }
  }

  if (isAnalyzing) {
    return <LoadingScreen />
  }

  const inputSections = [
    {
      title: "Identifiers",
      icon: "🛰️",
      fields: [
        { key: "kepid", label: "KepID", type: "text", placeholder: "e.g., 10797460" },
        { key: "koi_name", label: "KOI Name", type: "text", placeholder: "e.g., K00752.01" },
        { key: "kepler_name", label: "Kepler Name", type: "text", placeholder: "e.g., Kepler-227b" },
      ],
    },
    {
      title: "Transit & Orbit",
      icon: "🔄",
      fields: [
        { key: "orbital_period_days", label: "Orbital Period [days]", type: "text", placeholder: "e.g., 9.488..." },
        { key: "transit_epoch_bkjd", label: "Transit Epoch [BKJD]", type: "text", placeholder: "e.g., 170.538..." },
        { key: "impact_parameter", label: "Impact Parameter", type: "text", placeholder: "e.g., 0.146 (+.../-...)" },
        { key: "transit_duration_hrs", label: "Transit Duration [hrs]", type: "text", placeholder: "e.g., 2.95" },
        { key: "transit_depth_ppm", label: "Transit Depth [ppm]", type: "text", placeholder: "e.g., 615.8" },
        { key: "transit_snr", label: "Transit S/N", type: "text", placeholder: "e.g., 35.80" },
        { key: "tce_planet_number", label: "TCE Planet Number", type: "text", placeholder: "e.g., 1" },
        { key: "tce_delivery", label: "TCE Delivery", type: "text", placeholder: "e.g., q1_q17_dr25_tce" },
      ],
    },
    {
      title: "Planet Properties",
      icon: "🪐",
      fields: [
        { key: "planetary_radius_earth_radii", label: "Planetary Radius [Earth radii]", type: "text", placeholder: "e.g., 2.26 (+.../-...)" },
        { key: "equilibrium_temperature_k", label: "Equilibrium Temperature [K]", type: "text", placeholder: "e.g., 793" },
        { key: "insolation_flux_earth_flux", label: "Insolation Flux [Earth flux]", type: "text", placeholder: "e.g., 93.59 (+.../-...)" },
      ],
    },
    {
      title: "Stellar Properties",
      icon: "⭐",
      fields: [
        { key: "stellar_effective_temperature_k", label: "Stellar Effective Temperature [K]", type: "text", placeholder: "e.g., 5455±81" },
        { key: "stellar_surface_gravity_log", label: "Stellar Surface Gravity [log10(cm/s**2)]", type: "text", placeholder: "e.g., 4.467 (+.../-...)" },
        { key: "stellar_radius_solar_radii", label: "Stellar Radius [Solar radii]", type: "text", placeholder: "e.g., 0.927 (+.../-...)" },
      ],
    },
    {
      title: "Sky Position & Photometry",
      icon: "📍",
      fields: [
        { key: "ra_deg", label: "RA [deg]", type: "text", placeholder: "e.g., 291.934230" },
        { key: "dec_deg", label: "Dec [deg]", type: "text", placeholder: "e.g., 48.141651" },
        { key: "kepler_mag", label: "Kepler-band [mag]", type: "text", placeholder: "e.g., 15.347" },
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
          <div className="flex justify-center gap-4">
            <Button
              type="button"
              variant="outline"
              onClick={populateSampleData}
              className="gap-2"
            >
              <Sparkles className="w-4 h-4" />
              Fill with Sample Data (Kepler-442b)
            </Button>
          </div>
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

          {/* Extra card removed for new field set */}
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
