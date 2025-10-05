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
  name: string
  orbital_period: string
  planet_radius: string
  planet_mass: string
  semi_major_axis: string
  eccentricity: string
  inclination: string
  stellar_mass: string
  stellar_radius: string
  stellar_temperature: string
  stellar_luminosity: string
  distance_from_earth: string
  discovery_year: string
  detection_method: string
  equilibrium_temperature: string
  insolation_flux: string
  density: string
  surface_gravity: string
  escape_velocity: string
  albedo: string
  atmospheric_composition: string
  magnetic_field_strength: string
  rotation_period: string
  axial_tilt: string
  number_of_moons: string
  ring_system: string
  habitability_score: string
}

export default function NewPage() {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [formData, setFormData] = useState<PlanetFormData>({
    name: "",
    orbital_period: "",
    planet_radius: "",
    planet_mass: "",
    semi_major_axis: "",
    eccentricity: "",
    inclination: "",
    stellar_mass: "",
    stellar_radius: "",
    stellar_temperature: "",
    stellar_luminosity: "",
    distance_from_earth: "",
    discovery_year: "",
    detection_method: "",
    equilibrium_temperature: "",
    insolation_flux: "",
    density: "",
    surface_gravity: "",
    escape_velocity: "",
    albedo: "",
    atmospheric_composition: "",
    magnetic_field_strength: "",
    rotation_period: "",
    axial_tilt: "",
    number_of_moons: "",
    ring_system: "false",
    habitability_score: "",
  })

  const handleInputChange = (field: keyof PlanetFormData, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsAnalyzing(true)

    // Simulate AI analysis
    await new Promise((resolve) => setTimeout(resolve, 3000))

    // Navigate to results with form data
    const params = new URLSearchParams({
      planetData: JSON.stringify(formData),
      source: "new",
    })
    window.location.href = `/results?${params.toString()}`
  }

  if (isAnalyzing) {
    return <LoadingScreen />
  }

  const inputSections = [
    {
      title: "Basic Information",
      icon: "🌍",
      fields: [
        { key: "name", label: "Planet Name/Identifier", type: "text", placeholder: "e.g., Kepler-442b" },
        { key: "discovery_year", label: "Discovery Year", type: "number", placeholder: "e.g., 2015" },
        { key: "detection_method", label: "Detection Method", type: "text", placeholder: "e.g., Transit" },
        {
          key: "distance_from_earth",
          label: "Distance from Earth (light years)",
          type: "number",
          placeholder: "e.g., 1206",
        },
      ],
    },
    {
      title: "Orbital Properties",
      icon: "🔄",
      fields: [
        { key: "orbital_period", label: "Orbital Period (days)", type: "number", placeholder: "e.g., 112.3" },
        { key: "semi_major_axis", label: "Semi-Major Axis (AU)", type: "number", placeholder: "e.g., 0.409" },
        { key: "eccentricity", label: "Eccentricity", type: "number", placeholder: "e.g., 0.04" },
        { key: "inclination", label: "Inclination (degrees)", type: "number", placeholder: "e.g., 89.7" },
      ],
    },
    {
      title: "Physical Properties",
      icon: "⚡",
      fields: [
        { key: "planet_radius", label: "Planet Radius (Earth radii)", type: "number", placeholder: "e.g., 1.34" },
        { key: "planet_mass", label: "Planet Mass (Earth masses)", type: "number", placeholder: "e.g., 2.36" },
        { key: "density", label: "Density (g/cm³)", type: "number", placeholder: "e.g., 6.2" },
        { key: "surface_gravity", label: "Surface Gravity (m/s²)", type: "number", placeholder: "e.g., 13.2" },
        { key: "escape_velocity", label: "Escape Velocity (km/s)", type: "number", placeholder: "e.g., 16.8" },
      ],
    },
    {
      title: "Stellar Properties",
      icon: "⭐",
      fields: [
        { key: "stellar_mass", label: "Stellar Mass (Solar masses)", type: "number", placeholder: "e.g., 0.61" },
        { key: "stellar_radius", label: "Stellar Radius (Solar radii)", type: "number", placeholder: "e.g., 0.6" },
        { key: "stellar_temperature", label: "Stellar Temperature (K)", type: "number", placeholder: "e.g., 4402" },
        { key: "stellar_luminosity", label: "Stellar Luminosity (Solar)", type: "number", placeholder: "e.g., 0.17" },
      ],
    },
    {
      title: "Atmospheric & Climate",
      icon: "🌡️",
      fields: [
        {
          key: "equilibrium_temperature",
          label: "Equilibrium Temperature (K)",
          type: "number",
          placeholder: "e.g., 233",
        },
        { key: "insolation_flux", label: "Insolation Flux (Earth flux)", type: "number", placeholder: "e.g., 0.7" },
        { key: "albedo", label: "Albedo", type: "number", placeholder: "e.g., 0.3" },
        {
          key: "atmospheric_composition",
          label: "Atmospheric Composition",
          type: "text",
          placeholder: "e.g., Unknown",
        },
      ],
    },
    {
      title: "Additional Features",
      icon: "🔬",
      fields: [
        {
          key: "magnetic_field_strength",
          label: "Magnetic Field Strength (Earth = 1)",
          type: "number",
          placeholder: "e.g., 0.8",
        },
        { key: "rotation_period", label: "Rotation Period (days)", type: "number", placeholder: "e.g., 112.3" },
        { key: "axial_tilt", label: "Axial Tilt (degrees)", type: "number", placeholder: "e.g., 23.5" },
        { key: "number_of_moons", label: "Number of Moons", type: "number", placeholder: "e.g., 0" },
        { key: "habitability_score", label: "Habitability Score (0-1)", type: "number", placeholder: "e.g., 0.83" },
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
