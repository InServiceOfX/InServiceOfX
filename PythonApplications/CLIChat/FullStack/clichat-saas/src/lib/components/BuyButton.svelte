<script lang="ts">
  export let text = "Buy Now - $100"
  
  async function handleCheckout() {
    try {
      const response = await fetch('/api/create-checkout-session', {
        method: 'POST'
      })
      const { url } = await response.json()
      window.location.href = url
    } catch (err) {
      console.error('Error:', err)
    }
  }
</script>

<button 
  class="btn btn-primary" 
  on:click={handleCheckout}
>
  {text}
</button>
