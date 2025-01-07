import { PRIVATE_STRIPE_API_KEY } from "$env/static/private"
import { error, redirect } from "@sveltejs/kit"
import Stripe from "stripe"

const stripe = new Stripe(PRIVATE_STRIPE_API_KEY, { apiVersion: "2023-08-16" })
const DOWNLOAD_URL = "YOUR_DIRECT_DOWNLOAD_URL" // Replace with your actual download URL

export const load = async ({ url }) => {
  const sessionId = url.searchParams.get('session_id')
  if (!sessionId) {
    throw error(400, "Missing session ID")
  }

  try {
    const session = await stripe.checkout.sessions.retrieve(sessionId)
    
    if (session.payment_status !== 'paid') {
      throw error(400, "Payment not completed")
    }

    // Redirect to download
    redirect(303, DOWNLOAD_URL)
  } catch (err) {
    console.error("Error verifying payment:", err)
    throw error(500, "Could not verify payment")
  }
}