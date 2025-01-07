import { PRIVATE_STRIPE_API_KEY } from "$env/static/private"
import { error, json } from "@sveltejs/kit"
import Stripe from "stripe"

const stripe = new Stripe(PRIVATE_STRIPE_API_KEY, { apiVersion: "2023-08-16" })

export async function POST({ url }) {
  try {
    const session = await stripe.checkout.sessions.create({
      payment_method_types: ["card"],
      line_items: [
        {
          price_data: {
            currency: "usd",
            product_data: {
              name: "CLIChat download",
              description: "CLIChat download"
            },
            unit_amount: 10000, // $100.00 in cents
          },
          quantity: 1,
        },
      ],
      mode: "payment",
      success_url: `${url.origin}/payment/success?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${url.origin}/payment/cancel`,
      allow_promotion_codes: false
    })

    return json({ url: session.url })
  } catch (err) {
    console.error("Error creating checkout session:", err)
    throw error(500, "Could not create checkout session")
  }
}
