"use client"

import { Canvas } from "@react-three/fiber"
import { OrbitControls, Stars } from "@react-three/drei"
import { Planet } from "./planet"
import type { PlanetData } from "@/app/existing/page"

const PLANETS: PlanetData[] = [
  {
    id: "kepler-660b",
    name: "Kepler-660b",
    position: [-7, 3, -2],
    color: "#5ba3d0",
    size: 0.9,
    gemini_prompt: "Analyze the following light-curve feature summary and determine if it suggests an exoplanet transit. Provide a concise, human-readable explanation of the exoplanet with: a one-line assessment, a confidence (Low/Medium/High), and 3–5 bullet points highlighting notable features. Detected columns: time=TIME, flux=PDCSAP_FLUX Resampled length: 2048 Features (labeled): - mean: 0.999985 - std: 0.000505673 - min: 0.998047 - max: 1.00174 - median: 1 - q10: 0.999323 - q25: 0.999639 - q75: 1.00034 - q90: 1.00063 - dips_1sigma: 329 - dips_2sigma: 43 - dips_3sigma: 7 - peaks_1sigma: 324 - peaks_2sigma: 39 - transit_depth_1sigma: 0.000662359 - transit_depth_2sigma: 0.000346737 - transit_depth_min: 0.00193807 - variance: 2.55705e-07 - rmse: 0.000505673 - skewness: -0.145867 - kurtosis: 3.05632 - dominant_freq: 0.000976562 - dominant_power: 0.434151 - spectral_centroid: 0.178797 - low_freq_power: 6.31078 - mid_freq_power: 3.77863 - high_freq_power: 4.46777 - rolling_mean_mean: 0.999995 - rolling_mean_std: 0.000366151 - rolling_mean_min: 0.999241 - rolling_mean_max: 1.00075 - trend_slope: -6.90474e-09 - r_squared: 6.51679e-05 - autocorr_lag1: 3.91064e+06 - autocorr_lag5: 3.91065e+06 - autocorr_lag10: 3.91065e+06 - periodicity_strength: 8.009e+09",
    features: {
      orbital_period: 11.1,
      planet_radius: 1.67,
      planet_mass: 3.2,
      semi_major_axis: 0.08,
      eccentricity: 0.02,
      inclination: 89.5,
      stellar_mass: 0.7,
      stellar_radius: 0.65,
      stellar_temperature: 4200,
      stellar_luminosity: 0.2,
      distance_from_earth: 800,
      discovery_year: 2016,
      detection_method: "Transit",
      equilibrium_temperature: 326,
      insolation_flux: 1.1,
      density: 4.8,
      surface_gravity: 14.2,
      escape_velocity: 18.5,
      albedo: 0.4,
      atmospheric_composition: "Ocean world candidate",
      magnetic_field_strength: 1.0,
      rotation_period: 11.1,
      axial_tilt: 15,
      number_of_moons: 0,
      ring_system: false,
      habitability_score: 0.75,
    },
  },
  {
    id: "kepler-227b",
    name: "Kepler-227b",
    position: [4, -5, 1],
    color: "#f4a261",
    size: 1.1,
    gemini_prompt: "Analyze the following light-curve feature summary and determine if it suggests an exoplanet transit. Provide a concise, human-readable explanation of the exoplanet with: a one-line assessment, a confidence (Low/Medium/High), and 3–5 bullet points highlighting notable features. Detected columns: time=TIME, flux=PDCSAP_FLUX Resampled length: 2048 Features (labeled): - mean: 1.00014 - std: 0.00116202 - min: 0.997449 - max: 1.00387 - median: 0.999978 - q10: 0.998725 - q25: 0.999397 - q75: 1.00076 - q90: 1.00175 - dips_1sigma: 284 - dips_2sigma: 15 - dips_3sigma: 0 - peaks_1sigma: 354 - peaks_2sigma: 93 - transit_depth_1sigma: 0.00141728 - transit_depth_2sigma: 0.000744931 - transit_depth_min: 0.00269336 - variance: 1.35029e-06 - rmse: 0.00116202 - skewness: 0.512616 - kurtosis: 3.15558 - dominant_freq: 0.000488281 - dominant_power: 0.979019 - spectral_centroid: 0.146335 - low_freq_power: 10.7394 - mid_freq_power: 4.35787 - high_freq_power: 5.02915 - rolling_mean_mean: 1.00015 - rolling_mean_std: 0.00108019 - rolling_mean_min: 0.998174 - rolling_mean_max: 1.00282 - trend_slope: -5.61284e-08 - r_squared: 0.000815485 - autocorr_lag1: 740791 - autocorr_lag5: 740787 - autocorr_lag10: 740785 - periodicity_strength: 1.51714e+09",
    features: {
      orbital_period: 15.0,
      planet_radius: 2.1,
      planet_mass: 5.8,
      semi_major_axis: 0.12,
      eccentricity: 0.05,
      inclination: 88.2,
      stellar_mass: 0.85,
      stellar_radius: 0.8,
      stellar_temperature: 4800,
      stellar_luminosity: 0.35,
      distance_from_earth: 650,
      discovery_year: 2014,
      detection_method: "Transit",
      equilibrium_temperature: 415,
      insolation_flux: 2.2,
      density: 3.2,
      surface_gravity: 13.8,
      escape_velocity: 21.2,
      albedo: 0.25,
      atmospheric_composition: "Neptune-like atmosphere",
      magnetic_field_strength: 1.3,
      rotation_period: 15.0,
      axial_tilt: 12,
      number_of_moons: 2,
      ring_system: false,
      habitability_score: 0.45,
    },
  },
  {
    id: "kepler-664b",
    name: "Kepler-664b",
    position: [-3, 6, -4],
    color: "#e76f51",
    size: 0.6,
    gemini_prompt: "Analyze the following light-curve feature summary and determine if it suggests an exoplanet transit. Provide a concise, human-readable explanation of the exoplanet with: a one-line assessment, a confidence (Low/Medium/High), and 3–5 bullet points highlighting notable features. Detected columns: time=TIME, flux=PDCSAP_FLUX Resampled length: 2048 Features (labeled): - mean: 0.999855 - std: 0.00138774 - min: 0.995327 - max: 1.01925 - median: 1.00004 - q10: 0.998063 - q25: 0.99897 - q75: 1.00083 - q90: 1.00142 - dips_1sigma: 314 - dips_2sigma: 58 - dips_3sigma: 8 - peaks_1sigma: 287 - peaks_2sigma: 2 - transit_depth_1sigma: 0.00179267 - transit_depth_2sigma: 0.000885354 - transit_depth_min: 0.00452849 - variance: 1.92581e-06 - rmse: 0.00138774 - skewness: 0.815795 - kurtosis: 21.1899 - dominant_freq: 0.00244141 - dominant_power: 1.1072 - spectral_centroid: 0.198874 - low_freq_power: 12.902 - mid_freq_power: 6.34177 - high_freq_power: 12.5489 - rolling_mean_mean: 0.99985 - rolling_mean_std: 0.0011814 - rolling_mean_min: 0.996283 - rolling_mean_max: 1.00189 - trend_slope: 1.24286e-07 - r_squared: 0.00280354 - autocorr_lag1: 519111 - autocorr_lag5: 519111 - autocorr_lag10: 519111 - periodicity_strength: 1.06314e+09",
    features: {
      orbital_period: 25.0,
      planet_radius: 0.89,
      planet_mass: 0.8,
      semi_major_axis: 0.15,
      eccentricity: 0.03,
      inclination: 89.8,
      stellar_mass: 0.6,
      stellar_radius: 0.55,
      stellar_temperature: 4100,
      stellar_luminosity: 0.12,
      distance_from_earth: 900,
      discovery_year: 2015,
      detection_method: "Transit",
      equilibrium_temperature: 278,
      insolation_flux: 0.8,
      density: 6.1,
      surface_gravity: 10.2,
      escape_velocity: 13.8,
      albedo: 0.35,
      atmospheric_composition: "Rocky composition",
      magnetic_field_strength: 0.7,
      rotation_period: 25.0,
      axial_tilt: 18,
      number_of_moons: 0,
      ring_system: false,
      habitability_score: 0.82,
    },
  },
  {
    id: "kepler-228b",
    name: "Kepler-228b",
    position: [6, -2, 3],
    color: "#c44536",
    size: 1.3,
    gemini_prompt: "Analyze the following light-curve feature summary and determine if it suggests an exoplanet transit. Provide a concise, human-readable explanation of the exoplanet with: a one-line assessment, a confidence (Low/Medium/High), and 3–5 bullet points highlighting notable features. Detected columns: time=TIME, flux=PDCSAP_FLUX Resampled length: 2048 Features (labeled): - mean: 1.00005 - std: 0.0010514 - min: 0.996665 - max: 1.03695 - median: 1.00002 - q10: 0.999341 - q25: 0.999647 - q75: 1.00037 - q90: 1.00075 - dips_1sigma: 69 - dips_2sigma: 8 - dips_3sigma: 1 - peaks_1sigma: 86 - peaks_2sigma: 7 - transit_depth_1sigma: 0.000711879 - transit_depth_2sigma: 0.000405553 - transit_depth_min: 0.00338758 - variance: 1.10545e-06 - rmse: 0.0010514 - skewness: 21.7348 - kurtosis: 746.607 - dominant_freq: 0.00195312 - dominant_power: 0.220756 - spectral_centroid: 0.240927 - low_freq_power: 12.6685 - mid_freq_power: 10.4941 - high_freq_power: 21.3783 - rolling_mean_mean: 1.00006 - rolling_mean_std: 0.000293893 - rolling_mean_min: 0.999092 - rolling_mean_max: 1.00191 - trend_slope: 1.84027e-08 - r_squared: 0.000107078 - autocorr_lag1: 904705 - autocorr_lag5: 904706 - autocorr_lag10: 904706 - periodicity_strength: 1.85284e+09",
    features: {
      orbital_period: 8.0,
      planet_radius: 3.2,
      planet_mass: 12.5,
      semi_major_axis: 0.06,
      eccentricity: 0.0,
      inclination: 87.5,
      stellar_mass: 1.1,
      stellar_radius: 1.0,
      stellar_temperature: 5200,
      stellar_luminosity: 0.8,
      distance_from_earth: 450,
      discovery_year: 2013,
      detection_method: "Transit",
      equilibrium_temperature: 450,
      insolation_flux: 4.5,
      density: 2.8,
      surface_gravity: 15.2,
      escape_velocity: 25.8,
      albedo: 0.2,
      atmospheric_composition: "Eclipsing binary system",
      magnetic_field_strength: 2.0,
      rotation_period: 8.0,
      axial_tilt: 0,
      number_of_moons: 3,
      ring_system: true,
      habitability_score: 0.15,
    },
  },
  {
    id: "kepler-229b",
    name: "Kepler-229b",
    position: [-5, -3, -6],
    color: "#2a9d8f",
    size: 0.7,
    gemini_prompt: "Analyze the following light-curve feature summary and determine if it suggests an exoplanet transit. Provide a concise, human-readable explanation of the exoplanet with: a one-line assessment, a confidence (Low/Medium/High), and 3–5 bullet points highlighting notable features. Detected columns: time=TIME, flux=PDCSAP_FLUX Resampled length: 2048 Features (labeled): - mean: 1.00004 - std: 0.00122607 - min: 0.993651 - max: 1.00401 - median: 0.999997 - q10: 0.998636 - q25: 0.999256 - q75: 1.00074 - q90: 1.00148 - dips_1sigma: 281 - dips_2sigma: 24 - dips_3sigma: 15 - peaks_1sigma: 270 - peaks_2sigma: 82 - transit_depth_1sigma: 0.00140444 - transit_depth_2sigma: 0.000784089 - transit_depth_min: 0.00638902 - variance: 1.50324e-06 - rmse: 0.00122607 - skewness: -0.110464 - kurtosis: 5.39968 - dominant_freq: 0.000976562 - dominant_power: 1.1545 - spectral_centroid: 0.152251 - low_freq_power: 14.3452 - mid_freq_power: 5.94242 - high_freq_power: 6.81259 - rolling_mean_mean: 1.00004 - rolling_mean_std: 0.00106078 - rolling_mean_min: 0.997911 - rolling_mean_max: 1.0034 - trend_slope: -9.77599e-08 - r_squared: 0.00222215 - autocorr_lag1: 665285 - autocorr_lag5: 665283 - autocorr_lag10: 665278 - periodicity_strength: 1.3625e+09",
    features: {
      orbital_period: 6.1,
      planet_radius: 0.92,
      planet_mass: 0.62,
      semi_major_axis: 0.028,
      eccentricity: 0.005,
      inclination: 89.86,
      stellar_mass: 0.089,
      stellar_radius: 0.117,
      stellar_temperature: 2559,
      stellar_luminosity: 0.000553,
      distance_from_earth: 39.5,
      discovery_year: 2017,
      detection_method: "Transit",
      equilibrium_temperature: 251,
      insolation_flux: 0.66,
      density: 5.4,
      surface_gravity: 9.8,
      escape_velocity: 12.5,
      albedo: 0.3,
      atmospheric_composition: "Possible water vapor",
      magnetic_field_strength: 0.6,
      rotation_period: 6.1,
      axial_tilt: 0,
      number_of_moons: 0,
      ring_system: false,
      habitability_score: 0.85,
    },
  },
]

type GalaxyMapProps = {
  onPlanetSelect: (planet: PlanetData) => void
  selectedPlanet: PlanetData | null
}

export function GalaxyMap({ onPlanetSelect, selectedPlanet }: GalaxyMapProps) {
  return (
    <Canvas camera={{ position: [0, 0, 15], fov: 60 }}>
      <color attach="background" args={["#000000"]} />
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />

      {PLANETS.map((planet) => (
        <Planet
          key={planet.id}
          data={planet}
          onClick={() => onPlanetSelect(planet)}
          isSelected={selectedPlanet?.id === planet.id}
        />
      ))}

      <OrbitControls
        enableZoom={true}
        enablePan={true}
        enableRotate={true}
        minDistance={5}
        maxDistance={30}
        zoomSpeed={0.8}
      />
    </Canvas>
  )
}
