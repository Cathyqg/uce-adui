(function($, document) {
    'use strict';
    
    /**
     * Hero Banner component JavaScript
     */
    var HeroBanner = {
        
        selectors: {
            banner: '.hero-banner',
            cta: '.hero-banner__cta .btn',
            content: '.hero-banner__content'
        },
        
        init: function() {
            var self = this;
            
            $(document).ready(function() {
                self.bindEvents();
                self.initParallax();
                self.initLazyLoad();
            });
        },
        
        bindEvents: function() {
            var self = this;
            
            // CTA click tracking
            $(self.selectors.cta).on('click', function(e) {
                var bannerTitle = $(this).closest(self.selectors.banner)
                    .find('.hero-banner__title').text();
                
                // Analytics tracking
                if (typeof digitalData !== 'undefined') {
                    digitalData.event = digitalData.event || [];
                    digitalData.event.push({
                        eventInfo: {
                            eventName: 'hero-banner-cta-click',
                            eventAction: 'click',
                            eventLabel: bannerTitle
                        }
                    });
                }
            });
        },
        
        initParallax: function() {
            var self = this;
            var $banner = $(self.selectors.banner);
            
            if ($banner.length && !self.isMobile()) {
                $(window).on('scroll', function() {
                    var scrollPos = $(this).scrollTop();
                    $banner.css('background-position-y', scrollPos * 0.5 + 'px');
                });
            }
        },
        
        initLazyLoad: function() {
            var self = this;
            var $banners = $(self.selectors.banner);
            
            $banners.each(function() {
                var $banner = $(this);
                var bgImage = $banner.data('bg-image');
                
                if (bgImage) {
                    var img = new Image();
                    img.onload = function() {
                        $banner.css('background-image', 'url(' + bgImage + ')');
                        $banner.addClass('hero-banner--loaded');
                    };
                    img.src = bgImage;
                }
            });
        },
        
        isMobile: function() {
            return window.innerWidth < 768;
        }
    };
    
    HeroBanner.init();
    
})(jQuery, document);
