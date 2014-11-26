all: filter

filter: filter.o
	$(CC) $< -o $@ $(LDFLAGS) -lpthread -static

filter.o: filter.c
	$(CC) $< -c -o $@ $(CFLAGS)

filter.c: filter.C c.m4.linux
	m4 -s -Ulen -Uindex -Umkstemp c.m4.linux $< > $@

clean:
	-rm filter.c filter.o filter
