<script lang="ts">
  import { onMount } from 'svelte'
  import { Confetti } from 'svelte-confetti'
  
  export let data

  onMount(() => {
    if (data.success) {
      // Start download automatically
      window.location.href = data.downloadUrl
    }
  })
</script>

<div class="min-h-screen w-full px-3 py-5 sm:p-10">
  <div class="container mx-auto max-w-6xl text-center">
    <div class="mb-5 sm:mb-10">
      <svg class="checkmark mx-auto" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 52 52">
        <circle class="checkmark__circle" cx="26" cy="26" r="25" fill="none" />
        <path class="checkmark__check" fill="none" d="M14.1 27.2l7.1 7.2 16.7-16.8" />
      </svg>
    </div>

    <h1 class="mb-4">Thank You For Your Purchase!</h1>
    
    <p class="mb-5">Your download should begin automatically.</p>
    
    {#if data.downloadUrl}
      <p class="mt-4">
        If the download doesn't start, 
        <a href={data.downloadUrl} class="text-blue-600 underline">click here</a>
      </p>
    {/if}
  </div>
</div>

<div class="fixed inset-0 pointer-events-none">
  <Confetti
    x={[-5, 5]}
    y={[0, 0.1]}
    delay={[50, 2000]}
    duration="2000"
    amount="500"
    fallDistance="100vh"
  />
</div>