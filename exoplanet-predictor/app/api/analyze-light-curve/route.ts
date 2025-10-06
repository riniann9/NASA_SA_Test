import { NextResponse } from 'next/server'

export async function POST(req: Request) {
  try {
    const formData = await req.formData()
    const fitsFile = formData.get('fits_file') as File

    if (!fitsFile) {
      return new NextResponse(
        JSON.stringify({ error: 'No FITS file provided' }),
        { status: 400 }
      )
    }

    // Convert File to Buffer
    const buffer = Buffer.from(await fitsFile.arrayBuffer())
    
    // Create FormData to send to Python server
    const pythonFormData = new FormData()
    pythonFormData.append('file', new Blob([buffer]), fitsFile.name)

    // Call the Python server running on port 5000
    const pythonServerUrl = 'http://54.241.157.82:5000/api/extract'
    
    const response = await fetch(pythonServerUrl, {
      method: 'POST',
      body: pythonFormData,
    })

    if (!response.ok) {
      throw new Error(`Python server responded with status ${response.status}`)
    }

    const analysisResult = await response.json()
    return NextResponse.json(analysisResult)

  } catch (error: any) {
    console.error('Error in analyze-light-curve API:', error)
    return new NextResponse(
      JSON.stringify({ 
        error: error.message || 'Failed to analyze light curve',
        details: 'Make sure the Python server is running on port 5000'
      }),
      { status: 500 }
    )
  }
}