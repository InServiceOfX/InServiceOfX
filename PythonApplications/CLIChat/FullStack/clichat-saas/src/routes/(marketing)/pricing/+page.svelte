<script lang="ts">
  import PricingModule from "./pricing_module.svelte"
  import { WebsiteName } from "./../../../config"
  import BuyButton from '$lib/components/BuyButton.svelte'

  type PlanFeatureRow = {
    name: string
    freeIncluded?: boolean
    proIncluded?: boolean
    freeString?: string
    proString?: string
    header?: boolean
  }

  const planFeatures: PlanFeatureRow[] = [
    {
      name: "CLIChat",
      header: true,
    },
    {
      name: "Runs on Linux",
      freeIncluded: true,
      proIncluded: true,
    },
    {
      name: "Dynamically change system prompts on the fly",
      freeIncluded: true,
      proIncluded: true,
    },
    {
      name: "Saves your chat history *locally* (*NOT* in the \"cloud\")",
      freeIncluded: true,
      proIncluded: true,
    },
    {
      name: "Groq API support",
      freeIncluded: true,
      proIncluded: true,
    },
    {
      name: "Configurable",
      freeString: "yessir",
      proString: "yessir",
    },
  ]

  import XSmallIcon from '$lib/components/XSmallIcon.svelte'
  import InstagramSmallIcon from '$lib/components/InstagramSmallIcon.svelte'
</script>

<svelte:head>
  <title>Pricing</title>
  <meta name="description" content="Pricing - {WebsiteName}" />
</svelte:head>

<div class="min-h-[70vh] pb-8 pt-[5vh] px-4">
  <h1 class="text-3xl font-bold text-center">Pricing</h1>
  <h2 class="text-xl text-center text-slate-500 mt-1 pb-3">
    One-time purchase for a single, easy-to-install download
  </h2>

  <div class="w-full my-8">
    <PricingModule callToAction="Create an account or login to keep in touch (not needed to purchase)" highlightedPlanId="pro" />
    
    <div class="flex justify-center mt-12 mb-12">
      <BuyButton text="Purchase Now - $100" />
    </div>

    <h1 class="text-2xl font-bold text-center mt-24">Pricing FAQ</h1>
    <div class="flex place-content-center">
      <div class="join join-vertical max-w-xl py-6 mx-auto">
        <div class="collapse collapse-arrow join-item border border-primary">
          <input type="radio" name="faq-accordion" />
          <div class="collapse-title text-lg font-medium">
            Why you charge $100? Why you charge moneyz?
          </div>
          <div class="collapse-content">
            <p>Money directly goes to buying new computer equipment (GPU(s), I'd like to get a 
            new Mac to dev on and make Mac versions of my applications), and medical and vision 
            bills to keep me healthy and going. :/ :)</p>
          </div>
        </div>
        <div class="collapse collapse-arrow join-item border border-primary">
          <input type="radio" name="faq-accordion" />
          <div class="collapse-title text-lg font-medium">
            Windows or Mac support?
          </div>
          <div class="collapse-content">
            <p>
              Windows support coming soon. Help me get a Mac so I could provide Mac support by 
              purchasing 1 (or 2!) downloads!
            </p>
          </div>
        </div>
        <div class="collapse collapse-arrow join-item border border-primary">
          <input type="radio" name="faq-accordion" />
          <div class="collapse-title text-lg font-medium">
            Will there be new features in the future?
          </div>
          <div class="collapse-content">
            I plan to add new features including,
            <ul>
              <li>
                Memory (long-term, short-term)
              </li>
              <li>
                AI agents (!!)
              </li>
              <li>
                Other LLM API support
              </li>
            </ul>
            They will be bundled in a separate download installer. Depending on number and quality
            of features they can be offered free or at a discount to anyone that's bought 
            CLIChat before.
          </div>
        </div>
        <div class="collapse collapse-arrow join-item border border-primary">
          <input type="radio" name="faq-accordion" />
          <div class="collapse-title text-lg font-medium">
            Support, help?
          </div>
          <div class="collapse-content">
            <p>
              Reach out at
            </p>
            <p class="mt-4">
              <span class="inline-flex items-center gap-2">
                <XSmallIcon href="https://x.com/inserviceofx"/> 
                <InstagramSmallIcon href="https://www.instagram.com/inserviceofx/" />
              </span>
            </p>
          </div>
        </div>
      </div>
    </div>

    <svg style="display:none" version="2.0">
      <defs>
        <symbol
          id="checkcircle"
          viewBox="0 0 24 24"
          stroke-width="2"
          fill="currentColor"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path
            d="M16.417 10.283A7.917 7.917 0 1 1 8.5 2.366a7.916 7.916 0 0 1 7.917 7.917zm-4.105-4.498a.791.791 0 0 0-1.082.29l-3.828 6.63-1.733-2.08a.791.791 0 1 0-1.216 1.014l2.459 2.952a.792.792 0 0 0 .608.285.83.83 0 0 0 .068-.003.791.791 0 0 0 .618-.393L12.6 6.866a.791.791 0 0 0-.29-1.081z"
          />
        </symbol>
      </defs>
    </svg>

    <svg style="display:none" version="2.0">
      <defs>
        <symbol id="nocircle" viewBox="0 0 24 24" fill="currentColor">
          <path
            d="M12,2A10,10,0,1,0,22,12,10,10,0,0,0,12,2Zm4,11H8a1,1,0,0,1,0-2h8a1,1,0,0,1,0,2Z"
          />
        </symbol>
      </defs>
    </svg>

    <h1 class="text-2xl font-bold text-center mt-16">Clichat Features</h1>
    <h2 class="text-xl text-center text-slate-500 mt-1 pb-3">
     For the Linux-supported installer
    </h2>

    <div class="overflow-visible mx-auto max-w-xl mt-4">
      <table class="table">
        <thead
          class="text-lg sticky top-0 bg-base-100 bg-opacity-50 z-10 backdrop-blur"
        >
          <tr>
            <th></th>
            <th class="text-center">Can I Havez?</th>
          </tr>
        </thead>
        <tbody>
          {#each planFeatures as feature}
            {#if feature.header}
              <tr class="bg-base-200 font-bold">
                <td colspan="3">{feature.name} </td>
              </tr>
            {:else}
              <tr class="relative">
                <td>{feature.name} </td>
                <td class="text-center">
                  {#if feature.freeString}
                    {feature.freeString}
                  {:else if feature.freeIncluded}
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      class="w-8 h-8 ml-2 inline text-success"
                    >
                      <use href="#checkcircle" />
                    </svg>
                  {:else}
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      class="w-[26px] h-[26px] inline text-base-200"
                    >
                      <use href="#nocircle" />
                    </svg>
                  {/if}
                </td>
              </tr>
            {/if}
          {/each}
        </tbody>
      </table>
    </div>
  </div>
</div>
