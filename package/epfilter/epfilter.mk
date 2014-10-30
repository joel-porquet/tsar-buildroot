################################################################################
#
# epfilter
#
################################################################################

EPFILTER_VERSION = 1.0
EPFILTER_INSTALL_TARGET = YES
EPFILTER_SITE = $(TOPDIR)/package/epfilter/src
EPFILTER_SITE_METHOD = local
HOST_EPFILTER_DEPENDENCIES = host-m4

define EPFILTER_BUILD_CMDS
	$(TARGET_CONFIGURE_OPTS) $(MAKE) -C $(@D) all
endef

define EPFILTER_INSTALL_TARGET_CMDS
	$(INSTALL) -D -m 0755 $(@D)/filter $(TARGET_DIR)/usr/bin/epfilter
	$(INSTALL) -D -m 0644 $(@D)/philips_image.raw $(TARGET_DIR)/usr/share/epfilter/philips_image.raw
endef

define EPFILTER_PERMISSIONS
	/bin/epfilter  f  4755  0  0  -  -  -  -  -
endef

$(eval $(generic-package))
