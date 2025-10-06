// Generate descriptive query for planet image
export function generatePlanetImageQuery(planetData: any): string {
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
