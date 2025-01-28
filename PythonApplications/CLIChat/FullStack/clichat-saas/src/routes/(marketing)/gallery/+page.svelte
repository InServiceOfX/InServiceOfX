<script lang="ts">
  // Import any necessary components
  import { WebsiteName } from "./../../../config"

  // Define your gallery items
  const galleryItems = [
    {
      src: "/images/2025-01-21_07-39lifecoachsystempromptCLIChat.png",
      alt: "CLIChat Chat with Different System Prompts",
      title: "Chat while changing system prompts dynamically",
      description: "Use system prompt(s) to change the \"character\" of CLIChat"
    },
    {
      src: "/images/CartoonMiaoCatOnSimpleBackgroundInFront2FLUX.1-dev-Steps45Iter1Guidance6.5.png",
      alt: "CLIChat Mascot - Miao",
      title: "CLIchat - your command line Companion",
      description: "Your friendly command-line assistant, making terminal interactions more enjoyable"
    },
    {
      src: "/images/CartoonMiaoCatOnSimpleBackgroundInFront2FLUX.1-dev-Steps45Iter0Guidance5.5.png",
      alt: "CLIChat Mascot - Miao Relaxed",
      title: "CLIchat relaxed Mode",
      description: "Your command line companion in a relaxed pose, ready to assist"
    },
    {
      src: "/images/CartoonMiaoCatOnSimpleBackgroundInFront2FLUX.1-dev-Steps45Iter2Guidance7.5.png",
      alt: "CLIChat Mascot - Miao Active",
      title: "CLIchat Active Mode",
      description: "Your command line companion in action mode, eager to help"
    },
    {
      src: "/images/AIsystempromptcodingpeek_2.gif",
      alt: "CLIChat Coding Demo",
      title: "Coding Assistant Demo",
      description: "Watch CLIChat assist with coding tasks in real-time"
    },
    {
      src: "/images/AIagent800x480peek_2.gif",
      alt: "CLIChat chat prompt Demo",
      title: "Chat Prompt Demo",
      description: "See CLIChat's chat prompt capabilities in action"
    },
    {
      src: "/images/CartoonmiaocatSimplebackground0FLUX.1-dev-Steps30Iter1Guidance4WithTerminal.png",
      alt: "CLIChat Mascot - Terminal Guide",
      title: "Terminal Tips from the Terminal",
      description: "Your CLIchat companion showing you around the terminal environment"
    },
    {
      src: "/images/CartoonmiaocatSimplebackground0FLUX.1-dev-Steps30Iter9Guidance8WithTerminal.png",
      alt: "CLIChat Mascot - Command Line Expert",
      title: "Command Line Expertise",
      description: "Get expert guidance through your command line journey"
    }
  ]
</script>

<svelte:head>
  <title>Gallery - {WebsiteName}</title>
  <meta name="description" content="Visual showcase of CLIChat features and capabilities" />
</svelte:head>

<div class="container mx-auto px-4 py-8">
  <h1 class="text-4xl font-bold text-center mb-8">Gallery</h1>
  
  <!-- Static images section with masonry-like grid -->
  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
    {#each galleryItems.filter(item => !item.src.endsWith('.gif')) as item}
      <div class="card bg-base-100 shadow-xl {item.src.includes('WithTerminal') ? 'row-span-2' : ''}">
        <figure class="relative {item.src.includes('WithTerminal') ? 'h-[500px]' : 'h-[300px]'}">
          <img
            src={item.src}
            alt={item.alt}
            class="w-full h-full object-cover transform hover:scale-105 transition-duration-300"
          />
        </figure>
        <div class="card-body">
          <h2 class="card-title">{item.title}</h2>
          <p>{item.description}</p>
        </div>
      </div>
    {/each}
  </div>

  <!-- Optional: Add a section for animated GIFs -->
  <h2 class="text-3xl font-bold text-center mt-16 mb-8">Animations</h2>
  <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
    {#each galleryItems.filter(item => item.src.endsWith('.gif')) as gif}
      <div class="card bg-base-100 shadow-xl">
        <figure class="relative">
          <img
            src={gif.src}
            alt={gif.alt}
            class="w-full h-auto"
          />
        </figure>
        <div class="card-body">
          <h2 class="card-title">{gif.title}</h2>
          <p>{gif.description}</p>
        </div>
      </div>
    {/each}
  </div>
</div>

<style>
  figure img {
    transition: transform 0.3s ease;
  }
  
  figure:hover img {
    transform: scale(1.01);
  }
</style>