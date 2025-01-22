import { PRIVATE_STRIPE_API_KEY } from "$env/static/private"
import { error } from "@sveltejs/kit"
import Stripe from "stripe"
import { randomUUID } from 'crypto'

const stripe = new Stripe(PRIVATE_STRIPE_API_KEY, { apiVersion: "2023-08-16" })
const INSTALLER_FILE_NAME = "clichat-installer"

export const load = async ({ url, locals: { safeGetSession, supabaseServiceRole }, request }) => {
  const sessionId = url.searchParams.get('session_id')
  if (!sessionId) {
    throw error(400, "Missing session ID")
  }

  try {
    // Verify the Stripe payment
    const stripeSession = await stripe.checkout.sessions.retrieve(sessionId, {
      expand: ['payment_intent', 'customer']
    })
    
    if (stripeSession.payment_status !== 'paid') {
      throw error(400, "Payment not completed")
    }

    // Get user info - either logged in or from Stripe session
    const { session, user } = await safeGetSession()
    const customerEmail = stripeSession.customer_details?.email
    const customerName = `${stripeSession.customer_details?.name || 'Guest'}`

    // For guest users, store their data
    if (!user && stripeSession.customer) {
      const { data: existingCustomer } = await supabaseServiceRole
        .from('stripe_customers_guest')
        .select('stripe_customer_id')
        .eq('stripe_session_id', sessionId)
        .single()

      if (!existingCustomer) {
        const paymentIntent = stripeSession.payment_intent as Stripe.PaymentIntent
        
        await supabaseServiceRole
          .from('stripe_customers_guest')
          .insert({
            stripe_customer_id: stripeSession.customer.id,
            stripe_session_id: sessionId,
            customer_name: stripeSession.customer_details?.name,
            customer_email: stripeSession.customer_details?.email,
            customer_phone: stripeSession.customer_details?.phone,
            billing_address: stripeSession.customer_details?.address,
            payment_amount: stripeSession.amount_total,
            payment_currency: stripeSession.currency,
            payment_method: paymentIntent?.payment_method_types?.[0],
            payment_status: stripeSession.payment_status,
            updated_at: new Date().toISOString()
          })
      }
    }

    // Log the download attempt
    await supabaseServiceRole
      .from('download_logs')
      .insert({
        user_id: user?.id || 'guest',
        stripe_session_id: sessionId,
        file_name: INSTALLER_FILE_NAME,
        ip_address: request.headers.get('x-forwarded-for') || 'unknown',
        user_agent: request.headers.get('user-agent') || 'unknown',
        meta: { 
          payment_status: stripeSession.payment_status,
          customer: stripeSession.customer,
          customer_email: customerEmail,
          customer_name: customerName
        }
      })

    // Generate signed URL
    const { data: urlData, error: urlError } = await supabaseServiceRole
      .storage
      .from('installers')
      .createSignedUrl(INSTALLER_FILE_NAME, 300)

    if (urlError) {
      console.error('Storage URL error:', urlError)
      throw error(500, "Could not generate download URL")
    }

    return {
      success: true,
      downloadUrl: urlData.signedUrl,
      isGuest: !user,
      customerEmail,
      customerName
    }
  } catch (err) {
    console.error("Error processing success:", err)
    throw error(err.status || 500, err.message || "Could not process download")
  }
}