<script lang="ts">
  import {
    WebsiteName,
    WebsiteBaseUrl,
    WebsiteDescription,
  } from "./../../config"
  import BuyButton from '$lib/components/BuyButton.svelte'

  const ldJson = {
    "@context": "https://schema.org",
    "@type": "WebSite",
    name: WebsiteName,
    url: WebsiteBaseUrl,
  }
  const jsonldScript = `<script type="application/ld+json">${
    JSON.stringify(ldJson) + "<"
  }/script>`

  interface Feature {
    name: string
    description: string
    svgContent: string
    newPage?: boolean
    useBuyButton?: boolean
    link?: string
    linkText?: string
  }

  const features: Feature[] = [
    {
      name: "Pay once, no subscription",
      description:
        "You like setting your money on fire every month with subscriptions? I don't.\n\n" +        
        "I asked my co-workers how much they'd pay for a future AI app I dev. They wanted to NOT " +
        "pay a subscription. So pay once for the download, bring your API key, bring a \"free\" API " +
        "key, and chat away. No subscriptions, no cry.",
      svgContent: `<path d="M12 6V18" stroke="#1C274C" stroke-width="1.5" stroke-linecap="round"/>
<path d="M15 9.5C15 8.11929 13.6569 7 12 7C10.3431 7 9 8.11929 9 9.5C9 10.8807 10.3431 12 12 12C13.6569 12 15 13.1193 15 14.5C15 15.8807 13.6569 17 12 17C10.3431 17 9 15.8807 9 14.5" stroke="#1C274C" stroke-width="1.5" stroke-linecap="round"/>
<path d="M7 3.33782C8.47087 2.48697 10.1786 2 12 2C17.5228 2 22 6.47715 22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 10.1786 2.48697 8.47087 3.33782 7" stroke="#1C274C" stroke-width="1.5" stroke-linecap="round"/>`,
    },
    {
      name: "Command Line Prompts",
      description: "CLIChat lets you chat with your LLM in the command line of your OS's terminal " +
      "application. If you're hip to using your command line or terminal application, you'll like " +
      "CLIChat.",
      svgContent: `<path d="M2 12C2 8.22876 2 6.34315 3.17157 5.17157C4.34315 4 6.22876 4 10 4H14C17.7712 4 19.6569 4 20.8284 5.17157C22 6.34315 22 8.22876 22 12C22 15.7712 22 17.6569 20.8284 18.8284C19.6569 20 17.7712 20 14 20H10C6.22876 20 4.34315 20 3.17157 18.8284C2 17.6569 2 15.7712 2 12Z" stroke="#1C274C" stroke-width="1.5"/>
<path d="M10 16.5H6" stroke="#1C274C" stroke-width="1.5" stroke-linecap="round"/>
<path d="M8 13.5H6" stroke="#1C274C" stroke-width="1.5" stroke-linecap="round"/>
<path d="M2 10L22 10" stroke="#1C274C" stroke-width="1.5" stroke-linecap="round"/>
<path d="M14 15C14 14.0572 14 13.5858 14.2929 13.2929C14.5858 13 15.0572 13 16 13C16.9428 13 17.4142 13 17.7071 13.2929C18 13.5858 18 14.0572 18 15C18 15.9428 18 16.4142 17.7071 16.7071C17.4142 17 16.9428 17 16 17C15.0572 17 14.5858 17 14.2929 16.7071C14 16.4142 14 15.9428 14 15Z" stroke="#1C274C" stroke-width="1.5"/>`,
    },
    {
      name: "System prompts on the fly",
      description:
        "Configure your LLM chat with system prompts on the fly; combine two or more! Make it talk " +
        "back to you like your favorite cartoon character!",
      svgContent: `<circle cx="12" cy="12" r="3" stroke="#1C274C" stroke-width="1.5"/>
<path d="M13.7654 2.15224C13.3978 2 12.9319 2 12 2C11.0681 2 10.6022 2 10.2346 2.15224C9.74457 2.35523 9.35522 2.74458 9.15223 3.23463C9.05957 3.45834 9.0233 3.7185 9.00911 4.09799C8.98826 4.65568 8.70226 5.17189 8.21894 5.45093C7.73564 5.72996 7.14559 5.71954 6.65219 5.45876C6.31645 5.2813 6.07301 5.18262 5.83294 5.15102C5.30704 5.08178 4.77518 5.22429 4.35436 5.5472C4.03874 5.78938 3.80577 6.1929 3.33983 6.99993C2.87389 7.80697 2.64092 8.21048 2.58899 8.60491C2.51976 9.1308 2.66227 9.66266 2.98518 10.0835C3.13256 10.2756 3.3397 10.437 3.66119 10.639C4.1338 10.936 4.43789 11.4419 4.43786 12C4.43783 12.5581 4.13375 13.0639 3.66118 13.3608C3.33965 13.5629 3.13248 13.7244 2.98508 13.9165C2.66217 14.3373 2.51966 14.8691 2.5889 15.395C2.64082 15.7894 2.87379 16.193 3.33973 17C3.80568 17.807 4.03865 18.2106 4.35426 18.4527C4.77508 18.7756 5.30694 18.9181 5.83284 18.8489C6.07289 18.8173 6.31632 18.7186 6.65204 18.5412C7.14547 18.2804 7.73556 18.27 8.2189 18.549C8.70224 18.8281 8.98826 19.3443 9.00911 19.9021C9.02331 20.2815 9.05957 20.5417 9.15223 20.7654C9.35522 21.2554 9.74457 21.6448 10.2346 21.8478C10.6022 22 11.0681 22 12 22C12.9319 22 13.3978 22 13.7654 21.8478C14.2554 21.6448 14.6448 21.2554 14.8477 20.7654C14.9404 20.5417 14.9767 20.2815 14.9909 19.902C15.0117 19.3443 15.2977 18.8281 15.781 18.549C16.2643 18.2699 16.8544 18.2804 17.3479 18.5412C17.6836 18.7186 17.927 18.8172 18.167 18.8488C18.6929 18.9181 19.2248 18.7756 19.6456 18.4527C19.9612 18.2105 20.1942 17.807 20.6601 16.9999C21.1261 16.1929 21.3591 15.7894 21.411 15.395C21.4802 14.8691 21.3377 14.3372 21.0148 13.9164C20.8674 13.7243 20.6602 13.5628 20.3387 13.3608C19.8662 13.0639 19.5621 12.558 19.5621 11.9999C19.5621 11.4418 19.8662 10.9361 20.3387 10.6392C20.6603 10.4371 20.8675 10.2757 21.0149 10.0835C21.3378 9.66273 21.4803 9.13087 21.4111 8.60497C21.3592 8.21055 21.1262 7.80703 20.6602 7C20.1943 6.19297 19.9613 5.78945 19.6457 5.54727C19.2249 5.22436 18.693 5.08185 18.1671 5.15109C17.9271 5.18269 17.6837 5.28136 17.3479 5.4588C16.8545 5.71959 16.2644 5.73002 15.7811 5.45096C15.2977 5.17191 15.0117 4.65566 14.9909 4.09794C14.9767 3.71848 14.9404 3.45833 14.8477 3.23463C14.6448 2.74458 14.2554 2.35523 13.7654 2.15224Z" stroke="#1C274C" stroke-width="1.5"/>`,
    },
    {
      name: "Performance - lightweight",
      newPage: true,
      description:
        "I created CLIchat because I *did not* want to crash my local browser with another tab " +
        "and *did not* want to run bloat when all I want to do is to open up my terminal and " +
        "start chatting with a LLM.",
      svgContent: `<path d="M5.66953 9.91436L8.73167 5.77133C10.711 3.09327 11.7007 1.75425 12.6241 2.03721C13.5474 2.32018 13.5474 3.96249 13.5474 7.24712V7.55682C13.5474 8.74151 13.5474 9.33386 13.926 9.70541L13.946 9.72466C14.3327 10.0884 14.9492 10.0884 16.1822 10.0884C18.4011 10.0884 19.5106 10.0884 19.8855 10.7613C19.8917 10.7724 19.8977 10.7837 19.9036 10.795C20.2576 11.4784 19.6152 12.3475 18.3304 14.0857L15.2683 18.2287C13.2889 20.9067 12.2992 22.2458 11.3758 21.9628C10.4525 21.6798 10.4525 20.0375 10.4525 16.7528L10.4526 16.4433C10.4526 15.2585 10.4526 14.6662 10.074 14.2946L10.054 14.2754C9.6673 13.9117 9.05079 13.9117 7.81775 13.9117C5.59888 13.9117 4.48945 13.9117 4.1145 13.2387C4.10829 13.2276 4.10225 13.2164 4.09639 13.205C3.74244 12.5217 4.3848 11.6526 5.66953 9.91436Z" stroke="#1C274C" stroke-width="1.5"/>`,
    },
    {
      name: "Buy Now!",
      description: "Creating an account/login not required. If you want to have your purchase acknowledged so I " +
        "know and can send you news of future updates, create an account! Otherwise, Stripe does all " +
        "purchase fulfillment and your download starts immediately and is available for a limited " +
        "time.",
      svgContent: `<path d="M7.5 18C8.32843 18 9 18.6716 9 19.5C9 20.3284 8.32843 21 7.5 21C6.67157 21 6 20.3284 6 19.5C6 18.6716 6.67157 18 7.5 18Z" stroke="#1C274C" stroke-width="1.5"/>
<path d="M16.5 18.0001C17.3284 18.0001 18 18.6716 18 19.5001C18 20.3285 17.3284 21.0001 16.5 21.0001C15.6716 21.0001 15 20.3285 15 19.5001C15 18.6716 15.6716 18.0001 16.5 18.0001Z" stroke="#1C274C" stroke-width="1.5"/>
<path d="M2.26121 3.09184L2.50997 2.38429H2.50997L2.26121 3.09184ZM2.24876 2.29246C1.85799 2.15507 1.42984 2.36048 1.29246 2.75124C1.15507 3.14201 1.36048 3.57016 1.75124 3.70754L2.24876 2.29246ZM4.58584 4.32298L5.20507 3.89983V3.89983L4.58584 4.32298ZM5.88772 14.5862L5.34345 15.1022H5.34345L5.88772 14.5862ZM20.6578 9.88275L21.3923 10.0342L21.3933 10.0296L20.6578 9.88275ZM20.158 12.3075L20.8926 12.4589L20.158 12.3075ZM20.7345 6.69708L20.1401 7.15439L20.7345 6.69708ZM19.1336 15.0504L18.6598 14.469L19.1336 15.0504ZM5.70808 9.76V7.03836H4.20808V9.76H5.70808ZM2.50997 2.38429L2.24876 2.29246L1.75124 3.70754L2.01245 3.79938L2.50997 2.38429ZM10.9375 16.25H16.2404V14.75H10.9375V16.25ZM5.70808 7.03836C5.70808 6.3312 5.7091 5.7411 5.65719 5.26157C5.60346 4.76519 5.48705 4.31247 5.20507 3.89983L3.96661 4.74613C4.05687 4.87822 4.12657 5.05964 4.1659 5.42299C4.20706 5.8032 4.20808 6.29841 4.20808 7.03836H5.70808ZM2.01245 3.79938C2.68006 4.0341 3.11881 4.18965 3.44166 4.34806C3.74488 4.49684 3.87855 4.61727 3.96661 4.74613L5.20507 3.89983C4.92089 3.48397 4.54304 3.21763 4.10241 3.00143C3.68139 2.79485 3.14395 2.60719 2.50997 2.38429L2.01245 3.79938ZM4.20808 9.76C4.20808 11.2125 4.22171 12.2599 4.35876 13.0601C4.50508 13.9144 4.79722 14.5261 5.34345 15.1022L6.43198 14.0702C6.11182 13.7325 5.93913 13.4018 5.83723 12.8069C5.72607 12.1578 5.70808 11.249 5.70808 9.76H4.20808ZM10.9375 14.75C9.52069 14.75 8.53763 14.7482 7.79696 14.6432C7.08215 14.5418 6.70452 14.3576 6.43198 14.0702L5.34345 15.1022C5.93731 15.7286 6.69012 16.0013 7.58636 16.1283C8.45674 16.2518 9.56535 16.25 10.9375 16.25V14.75ZM4.95808 6.87H17.0888V5.37H4.95808V6.87ZM19.9232 9.73135L19.4235 12.1561L20.8926 12.4589L21.3923 10.0342L19.9232 9.73135ZM17.0888 6.87C17.9452 6.87 18.6989 6.871 19.2937 6.93749C19.5893 6.97053 19.8105 7.01643 19.9659 7.07105C20.1273 7.12776 20.153 7.17127 20.1401 7.15439L21.329 6.23978C21.094 5.93436 20.7636 5.76145 20.4632 5.65587C20.1567 5.54818 19.8101 5.48587 19.4604 5.44678C18.7646 5.369 17.9174 5.37 17.0888 5.37V6.87ZM21.3933 10.0296C21.5625 9.18167 21.7062 8.47024 21.7414 7.90038C21.7775 7.31418 21.7108 6.73617 21.329 6.23978L20.1401 7.15439C20.2021 7.23508 20.2706 7.38037 20.2442 7.80797C20.2168 8.25191 20.1002 8.84478 19.9223 9.73595L21.3933 10.0296ZM16.2404 16.25C17.0021 16.25 17.6413 16.2513 18.1566 16.1882C18.6923 16.1227 19.1809 15.9794 19.6074 15.6318L18.6598 14.469C18.5346 14.571 18.3571 14.6525 17.9744 14.6994C17.5712 14.7487 17.0397 14.75 16.2404 14.75V16.25ZM19.4235 12.1561C19.2621 12.9389 19.1535 13.4593 19.0238 13.8442C18.9007 14.2095 18.785 14.367 18.6598 14.469L19.6074 15.6318C20.0339 15.2842 20.2729 14.8346 20.4453 14.3232C20.6111 13.8312 20.7388 13.2049 20.8926 12.4589L19.4235 12.1561Z" fill="#1C274C"/>`,
      useBuyButton: true
    },
    {
      name: "OS Support: Linux",
      description: "Linux support right now only. Tested on Debian. I'm hoping to add Windows " +
        "support soon. Please, please, please (Sabrina Carp.) buy a download (buy 2!) so I can buy a " +
        "Mac to dev on for Mac support.",
      svgContent: `<path d="M12.004 2c-1.542.001-2.423.593-2.936 1.493-.512.9-.624 2.108-.624 3.427v1.825c0 .454-.216.895-.477 1.293-.261.398-.6.774-.937 1.149l-.301.338c-.89.998-1.752 1.967-2.217 3.116-.465 1.15-.57 2.506.264 3.795.558.862 1.336 1.368 2.15 1.614.815.246 1.706.252 2.615.187.545-.039 1.054-.11 1.459-.17v.016c.112-.016.22-.032.322-.047 1.892-.269 3.763-.346 5.642.32.916.325 1.727.218 2.322-.122.595-.34.982-.917 1.235-1.486.417-.94.461-1.88.225-2.564-.236-.685-.725-1.186-1.179-1.655-.454-.47-.887-.93-.978-1.587-.032-.233-.02-.487.017-.752.037-.265.098-.541.16-.817l.02-.093c.147-.677.292-1.353.158-1.97-.134-.616-.547-1.17-1.514-1.467-.484-.148-.875-.37-1.133-.63-.258-.26-.397-.548-.432-.854-.07-.614.144-1.305.36-1.993.11-.355.223-.708.298-1.06.074-.35.119-.708.044-1.03-.18-.77-.83-1.324-1.523-1.642-.693-.319-1.436-.4-1.952-.408-.043 0-.085-.001-.126-.001zm.031 1c.033 0 .067 0 .102.002.458.006 1.077.082 1.614.333.537.25.931.615 1.03 1.051.034.144.02.378-.043.676-.064.298-.17.63-.282.992-.225.72-.466 1.516-.37 2.33.066.574.317 1.104.724 1.516.407.413.935.684 1.486.852.524.16.743.398.83.78.088.382-.028.91-.17 1.566l-.021.093c-.064.287-.13.584-.172.879-.042.295-.068.62-.018.967.137.985.77 1.65 1.265 2.162.414.428.78.81.934 1.242.154.432.126 1.065-.198 1.79-.187.421-.453.797-.79.982-.338.184-.766.242-1.437-.003-2.113-.75-4.146-.66-6.14-.378-.105.015-.216.032-.33.048-.406.06-.897.128-1.407.164-.85.06-1.631.05-2.29-.149-.66-.199-1.213-.576-1.644-1.24-.613-.947-.564-1.901-.194-2.79.37-.888 1.086-1.713 1.982-2.718l.302-.338c.35-.392.726-.806 1.038-1.281.312-.474.65-1.099.65-1.847V6.92c0-1.254.113-2.319.529-3.055C8.765 3.128 9.34 3 10.635 3c.467 0 .88.332 1.4.332z" fill="#1C274C"/>
    <path d="M9.5 7a1 1 0 1 0 0-2 1 1 0 0 0 0 2z" fill="#1C274C"/>
    <path d="M14.5 7a1 1 0 1 0 0-2 1 1 0 0 0 0 2z" fill="#1C274C"/>`,
    }
  ]

  import XSmallIcon from '$lib/components/XSmallIcon.svelte'
  import InstagramSmallIcon from '$lib/components/InstagramSmallIcon.svelte'
</script>

<svelte:head>
  <title>{WebsiteName}</title>
  <meta name="description" content={WebsiteDescription} />
  <!-- eslint-disable-next-line svelte/no-at-html-tags -->
  {@html jsonldScript}
</svelte:head>

<div class="hero min-h-[60vh]">
  <div class="hero-content text-center py-12">
    <div class="max-w-xl">
      <div
        class="text-xl md:text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-accent mb-3 md:mb-7 pb-1"
      >
        CLIChat - Command Line Interface for AI-powered Chat
      </div>

      <div class="mb-8">
        <img
          src="/images/2025-01-21_07-39lifecoachsystempromptCLIChat.png"
          alt="CLIChat Screenshot"
          class="w-full max-w-2xl mx-auto rounded-lg shadow-lg"
        />
      </div>

      <div
        class="text-4xl md:text-6xl font-bold px-2"
        style="line-height: 1.2;"
      >
        The
        <span
          class="underline decoration-secondary decoration-4 md:decoration-[6px]"
          >fast</span
        >, and
        <span
          class="underline decoration-secondary decoration-4 md:decoration-[6px]"
          >light-weight</span
        >
        <span> AI-powered chat right in your
        <span class="underline decoration-secondary decoration-4 md:decoration-[6px]">
        <em>command-line terminal</em></span></span>
      </div>
      <div class="mt-6 md:mt-10 text-sm md:text-lg">
        Built with <a
          href="https://python-prompt-toolkit.readthedocs.io/en/master/"
          class="link font-bold"
          target="_blank">Python Prompt Toolkit</a
        >, and
        <a href="https://docs.pytest.org/en/stable/" class="link font-bold" target="_blank"
          >pytest unit testing</a
        >
      </div>
      <div class="mt-6 md:mt-2">
        <a
          href="/blog/quick_getting_started"
        >
          <button class="btn btn-outline btn-primary btn-sm px-6 mt-3 mx-2"
            >Read the Docs</button
          >
        </a>
      </div>
      <div>
        <BuyButton text="Purchase Now - $100" />
      </div>
    </div>
  </div>
</div>

<div class="flex justify-center items-center py-8 px-4">
  <div class="max-w-5xl mx-auto flex flex-col md:flex-row items-center gap-8">
    <div class="md:w-[55%]">
      <img
        src="/images/CartoonMiaoCatOnSimpleBackgroundInFront2FLUX.1-dev-Steps45Iter0Guidance5.5.png"
        alt="Cartoon Cat Mascot - Relaxed Pose"
        class="w-full max-w-md mx-auto transform hover:scale-105 transition-transform duration-300"
        style="filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1))"
      />
    </div>
    <div class="md:w-[45%]">
      <img
        src="/images/CartoonMiaoCatOnSimpleBackgroundInFront2FLUX.1-dev-Steps45Iter2Guidance7.5.png"
        alt="Cartoon Cat Mascot - Active Pose"
        class="w-full max-w-sm mx-auto transform hover:scale-105 transition-transform duration-300"
        style="filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1))"
      />
    </div>
  </div>
</div>

<!-- Add the animated GIF here -->
<div class="max-w-4xl mx-auto px-4 py-12">
  <div class="mockup-browser border">
    <div class="mockup-browser-toolbar">
      <div class="input" style="background:#eee;">
        Watch CLIChat in action
      </div>
    </div>
    <div class="flex justify-center">
      <img
        src="/images/AIsystempromptcodingpeek_2.gif"
        alt="CLIChat Demo Animation"
        class="w-full h-auto object-contain"
      />
    </div>
  </div>
</div>

<div class="min-h-[60vh]">
  <div class="pt-20 pb-8 px-7">
    <div class="max-w-lg mx-auto text-center">
      <div
        class="text-3xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-accent pb-2"
      >
        Explore the Features
      </div>
      <div class="mt-4 text-xl font-bold">
        And learn more about using a LLM chat
        <span
          class="underline decoration-secondary decoration-[3px] md:decoration-[4px]"
        >
          directly in your command line
        </span><br>
        (and yes, part of what it is is a
        <span
          class="underline decoration-secondary decoration-[3px] md:decoration-[4px]"
        >LLM wrapper</span>, let's get that out of the way.)
      </div>
    </div>

    <div
      class="flex gap-6 mt-12 max-w-[1064px] mx-auto place-content-center flex-wrap"
    >
      {#each features as feature}
        <div class="card bg-white w-[270px] min-h-[300px] flex-none shadow-xl">
          <div class="card-body items-center text-center p-[24px] pt-[32px]">
            <div>
              <svg
                width="50px"
                height="50px"
                class="mb-2 mt-1"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <!-- eslint-disable-next-line svelte/no-at-html-tags -->
                {@html feature.svgContent}
              </svg>
            </div>
            <h2 class="card-title">
              {feature.name}
            </h2>
            <p class="text-sm">
              {feature.description}
            </p>
            {#if feature.useBuyButton}
              <div class="pb-4">
                <BuyButton />
              </div>
            {:else if feature.link}
              <a
                href={feature.link}
                class="pb-4"
                target={feature.newPage ? "_blank" : ""}
              >
                <button
                  class="btn btn-xs btn-outline rounded-full btn-primary min-w-[100px]"
                  >{feature.linkText ? feature.linkText : "Try It"}</button
                >
              </a>
            {/if}
          </div>
        </div>
      {/each}
    </div>
  </div>
</div>

<div class="flex justify-center items-center py-8">
  <div class="max-w-xl mx-auto">
    <img
      src="/images/CartoonMiaoCatOnSimpleBackgroundInFront2FLUX.1-dev-Steps45Iter1Guidance6.5.png"
      alt="Cartoon Cat Mascot - Playful Pose"
      class="w-4/5 mx-auto transform hover:scale-105 transition-transform duration-300"
      style="filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1))"
    />
  </div>
</div>

<div class="hero mt-8">
  <div class="hero-content text-center pb-16 pt-4 px-4">
    <div class="max-w-lg">
      <div
        class="text-3xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-accent mt-4 pb-2"
      >
        Buy it, Download it, and Start Chatting
      </div>
      <div
        class="flex flex-col lg:flex-row mt-6 gap-6 place-content-center content-center"
      >
        <div class="hidden md:block">
          <a href="https://x.com/inserviceofx" target="_blank" class="link">
            <div class="mockup-browser border">
              <div class="mockup-browser-toolbar">
                <div class="input" style="background:#eee;">
                  Not in your browser, but in your command line!
                </div>
              </div>
              <div class="flex justify-center">
                <img
                  alt="CLIChat Demo Animation"
                  class="aspect-[2044/1242] w-full h-auto object-contain"
                  src="/images/AIagent800x480peek_2.gif"
                />
              </div>
            </div>
          </a>
        </div>
        <div class="md:hidden">
          <a href="https://criticalmoments.io" target="_blank" class="link">
            <div class="card shadow-xl border overflow-hidden">
              <img
                alt="Screenshot of criticalmoments.io homepage"
                class="aspect-[2044/1242]"
                src="/images/example-home.png"
              />
            </div></a
          >
        </div>
        <div class="min-w-[270px] lg:min-w-[420px] flex mt-6 lg:mt-0">
          <div class="my-auto">
            <div class="px-4 text-lg md:text-xl">
CLIChat was created by <span
                  class="font-bold whitespace-nowrap">In Service of X</span
                > as a lightweight and 
                <span class="underline decoration-secondary decoration-[3px]"
                  >low memory footprint UX </span
                > (user interface) to avoid browser bloat and subscription dependence.
            </div>
            <div class="px-4 mt-6 text-lg md:text-xl">
              Keep in touch, reach out, and follow at 
              <span class="inline-flex items-center gap-2">
                <XSmallIcon href="https://x.com/inserviceofx"/> 
                <InstagramSmallIcon href="https://www.instagram.com/inserviceofx/" />
              </span>
            </div>
            <div class="mt-4 text-large">
                <BuyButton text="Purchase Now - $100" />
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="flex justify-center items-center py-8 px-4">
  <div class="max-w-3xl mx-auto">
    <img
      src="/images/CartoonmiaocatSimplebackground0FLUX.1-dev-Steps30Iter9Guidance8WithTerminal.png"
      alt="CLIChat Mascot - Terminal Expert"
      class="w-full max-w-xl mx-auto transform hover:scale-105 transition-transform duration-300"
      style="filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1))"
    />
  </div>
</div>